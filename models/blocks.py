import torch
import torch.nn as nn
import math
import torch.nn.functional as F
try:
    # If you installed the official mamba_ssm package
    from mamba_ssm import Mamba
except ImportError:
    # In case the Voxel-Mamba repo vendors its own Mamba implementation,
    # you can import from there instead (adjust path to your local repo):
    # from Voxel-Mamba.mamba.mamba_ssm import Mamba
    raise ImportError(
        "Could not import Mamba. Install mamba-ssm or import from Voxel-Mamba repo."
    )

from .hilbert_small import hilbert_order_3d

def chunkify_3d(x: torch.Tensor, patch_size: int):
    """
    x: [B, C, D, H, W], cubic volume with D == H == W
    patch_size: R, must divide D
    returns:
        patches: [B, N3, C, R, R, R]
        where N = D // R, N3 = N**3
    """
    B, C, D, H, W = x.shape
    assert D == H == W, "Assume cubic grid"
    assert D % patch_size == 0, "patch_size must divide D"

    R = patch_size
    N = D // R  # chunks per axis

    # [B, C, N, R, N, R, N, R]
    x = x.view(B, C, N, R, N, R, N, R)

    # [B, N, N, N, C, R, R, R]
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)

    # [B, N^3, C, R, R, R]
    patches = x.reshape(B, N**3, C, R, R, R)
    return patches

def unchunkify_3d(patches: torch.Tensor, patch_size: int, full_size: int):
    """
    patches: [B, N3, C, R, R, R]
    patch_size: R
    full_size: D (e.g., 32)
    returns:
        x: [B, C, D, D, D]
    """
    B, N3, C, R, _, _ = patches.shape
    D = full_size
    N = D // R
    assert N**3 == N3

    # [B, N, N, N, C, R, R, R]
    x = patches.view(B, N, N, N, C, R, R, R)

    # [B, C, N, R, N, R, N, R]
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)

    # [B, C, D, D, D]
    x = x.reshape(B, C, D, D, D)
    return x

class BasicBlock(nn.Module):
    def __init__(self,
                 mode: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 1,
                 output_padding: int = 1,
                 leaky: bool = False,
                 dropout: float = 0.0):
        super(BasicBlock, self).__init__()

        if mode == 'Encoder':
            self._conv1 = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding
            )
        elif mode == 'Decoder':
            self._conv1 = nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride,
                padding=padding, output_padding=output_padding
            )
        else:
            raise ValueError("Wrong mode, please enter 'Encoder' or 'Decoder'.")

        self._conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self._conv1 = nn.utils.parametrizations.weight_norm(self._conv1, name='weight')
        self._conv2 = nn.utils.parametrizations.weight_norm(self._conv2, name='weight')

        self._dropout = nn.Dropout3d(p=dropout) if dropout > 0 else None
        self._relu = nn.LeakyReLU(0.2) if leaky else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self._conv1(x)
        out = self._relu(out)

        identity = out

        out = self._conv2(out)
        out = self._relu(out)

        if self._dropout is not None:
            out = self._dropout(out)

        out = out + identity
        out = self._relu(out)

        return out

class ImplicitWindowEmbedding3D(nn.Module):
    """
    3D version of IWP/IWE:
      - Given grid size (D, H, W) and window size (wd, wh, ww)
      - For each voxel (z,y,x) compute:
          [ win_z, win_y, win_x, local_z, local_y, local_x ]
      - Run through small MLP -> [C] and add to features.
    """

    def __init__(self, channels: int, grid_size, window_size):
        super().__init__()
        D, H, W = grid_size
        wd, wh, ww = window_size

        self.channels = channels
        self.grid_size = grid_size
        self.window_size = window_size

        # Simple 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(6, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

        # Precompute coordinates & window indices for this grid size
        z, y, x = torch.meshgrid(
            torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij"
        )  # each [D,H,W]

        win_z = z // wd
        win_y = y // wh
        win_x = x // ww

        local_z = z % wd
        local_y = y % wh
        local_x = x % ww

        pos = torch.stack(
            [win_z, win_y, win_x, local_z, local_y, local_x], dim=-1
        )  # [D,H,W,6]

        pos = pos.float()

        # --- SAFE NORMALIZATION FOR WINDOW IDS ---
        max_ids = pos[..., 0:3].amax()
        if max_ids > 0:
            pos[..., 0:3] = pos[..., 0:3] / max_ids  # window ids in [0,1]
        # else: all window ids are 0 → leave as 0, no normalization

        # --- NORMALIZATION FOR LOCAL OFFSETS ---
        denom = torch.tensor(
            [max(wd - 1, 1), max(wh - 1, 1), max(ww - 1, 1)],
            dtype=torch.float32,
        )
        pos[..., 3:6] = pos[..., 3:6] / denom


        self.register_buffer("pos_features", pos)  # [D,H,W,6]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, D, H, W]
        returns: x + IWE, same shape
        """
        B, C, D, H, W = x.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"
        assert (D, H, W) == self.grid_size

        pos = self.pos_features.view(-1, 6)  # [N,6]
        emb = self.mlp(pos)                  # [N,C]
        emb = emb.view(1, D, H, W, C).permute(0, 4, 1, 2, 3)  # [1,C,D,H,W]

        return x + emb  # broadcast over batch

class MultiScaleChunkVoxelMamba3D(nn.Module):
    """
    Multi-scale Mamba block:
      - F32: voxel-level Mamba on full 32^3 grid (with IWE).
      - F8:  chunk-level Mamba with chunk_size = 8.
      - F4:  chunk-level Mamba with chunk_size = 4.
      - Fusion: Fe = F32 + F8 + F4 + x.

    Optional:
      - Memory bank on chunk tokens before chunk-Mamba.

    Input / Output: [B, C, D, H, W] with D == H == W == grid_size[0] (e.g., 32).
    """

    def __init__(
        self,
        channels: int,
        grid_size=(32, 32, 32),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        window_size=(4, 4, 4),    # for voxel-level IWE
        chunk_sizes=(8, 4),       # chunk branches

        use_memory_bank: bool = False,
        memory_num_slots: int = 64,
        memory_use_proj: bool = True,
        memory_share_between_branches: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size

        self.use_memory_bank = use_memory_bank
        self.memory_num_slots = memory_num_slots
        self.memory_use_proj = memory_use_proj
        self.memory_share_between_branches = memory_share_between_branches

        D, H, W = grid_size
        assert D == H == W, "Assume cubic grid"
        self.D = D

        # -------------------------
        # 1) Voxel-level branch (F32)
        # -------------------------
        self.ln_32 = nn.LayerNorm(channels)
        self.mamba_32 = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.iwe_32 = ImplicitWindowEmbedding3D(
            channels=channels,
            grid_size=grid_size,
            window_size=window_size,
        )
        order_32 = hilbert_order_3d(D, H, W)
        self.register_buffer("seq_order_32", order_32)

        # -------------------------
        # 2) Chunk branches (F8, F4)
        # -------------------------
        assert chunk_sizes == (8, 4) or chunk_sizes == (4, 8) or len(chunk_sizes) == 2
        R_a, R_b = chunk_sizes

        # ---- R_a branch ----
        self.R_a = R_a
        N_a = D // R_a
        self.N_a = N_a
        self.num_chunks_a = N_a ** 3

        chunk_dim_a = channels * (R_a ** 3)
        self.embed_a = nn.Linear(chunk_dim_a, channels)
        self.unembed_a = nn.Linear(channels, chunk_dim_a)

        self.ln_a = nn.LayerNorm(channels)
        self.mamba_a = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        order_chunks_a = hilbert_order_3d(N_a, N_a, N_a)
        self.register_buffer("chunk_order_a", order_chunks_a)

        # ---- R_b branch ----
        self.R_b = R_b
        N_b = D // R_b
        self.N_b = N_b
        self.num_chunks_b = N_b ** 3

        chunk_dim_b = channels * (R_b ** 3)
        self.embed_b = nn.Linear(chunk_dim_b, channels)
        self.unembed_b = nn.Linear(channels, chunk_dim_b)

        self.ln_b = nn.LayerNorm(channels)
        self.mamba_b = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        order_chunks_b = hilbert_order_3d(N_b, N_b, N_b)
        self.register_buffer("chunk_order_b", order_chunks_b)

        # -------------------------
        # 3) Optional memory bank(s)
        # -------------------------
        if self.use_memory_bank:
            if self.memory_share_between_branches:
                shared_bank = MemoryBank3D(
                    dim=channels,
                    num_slots=memory_num_slots,
                    use_proj=memory_use_proj,
                )
                self.memory_bank_a = shared_bank
                self.memory_bank_b = shared_bank
            else:
                self.memory_bank_a = MemoryBank3D(
                    dim=channels,
                    num_slots=memory_num_slots,
                    use_proj=memory_use_proj,
                )
                self.memory_bank_b = MemoryBank3D(
                    dim=channels,
                    num_slots=memory_num_slots,
                    use_proj=memory_use_proj,
                )
        else:
            self.memory_bank_a = None
            self.memory_bank_b = None

    # ---------- helpers ----------
    def _apply_mamba_voxel(self, x, seq_order, ln_layer, mamba_module):
        """
        x: [B, C, D, H, W]
        seq_order: [N] (Hilbert over all voxels)
        returns: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        N = D * H * W

        x_flat = x.view(B, C, N).transpose(1, 2)    # [B, N, C]
        x_flat = x_flat[:, seq_order, :]            # reorder

        x_flat = ln_layer(x_flat)                   # [B, N, C]
        y_flat = mamba_module(x_flat)               # [B, N, C]

        inv = torch.argsort(seq_order)
        y_flat = y_flat[:, inv, :]                  # [B, N, C]

        y = y_flat.transpose(1, 2).view(B, C, D, H, W)
        return y

    def _apply_mamba_chunks(
        self,
        x,
        R,
        N,
        num_chunks,
        embed,
        unembed,
        ln_layer,
        mamba_module,
        chunk_order,
        memory_bank=None,
        return_attn: bool = False,
    ):
        """
        x: [B, C, D, H, W]
        R: chunk size (4 or 8)
        N: D // R
        num_chunks: N^3
        returns: [B, C, D, H, W] branch output
        """
        B, C, D, H, W = x.shape

        # 1) chunkify: [B, num_chunks, C, R,R,R]
        patches = chunkify_3d(x, patch_size=R)
        B_, N3, C_, _, _, _ = patches.shape
        assert B_ == B and C_ == C
        assert N3 == num_chunks

        # 2) flatten chunks and embed
        chunks_flat = patches.view(B_, N3, -1)        # [B, N^3, C*R^3]
        tokens = embed(chunks_flat)                   # [B, N^3, C]

        attn = None

        # 2.5) optional memory-bank retrieval
        if memory_bank is not None:
            tokens, attn = memory_bank(tokens)        # [B, N^3, C], [B, N^3, M]

        # 3) Hilbert order over chunk grid
        order = chunk_order                           # [N^3]
        tokens = tokens[:, order, :]                  # [B, N^3, C]

        # 4) LN + Mamba
        tokens = ln_layer(tokens)                     # [B, N^3, C]
        tokens = mamba_module(tokens)                 # [B, N^3, C]

        # 5) undo Hilbert
        inv = torch.argsort(order)
        tokens = tokens[:, inv, :]                    # [B, N^3, C]

        # 6) decode back to chunks
        chunks_flat_out = unembed(tokens)             # [B, N^3, C*R^3]
        patches_out = chunks_flat_out.view(B_, N3, C_, R, R, R)

        # 7) unchunkify to full volume
        y = unchunkify_3d(patches_out, patch_size=R, full_size=D)  # [B,C,D,D,D]

        if return_attn:
            return y, attn
        return y

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, return_memory_attn: bool = False):
        """
        x: [B, C, D, H, W] with D==H==W==self.D

        if return_memory_attn=True and memory is enabled:
            returns:
                Fe, aux
            where aux is a dict with optional attention maps
        """
        B, C, D, H, W = x.shape
        assert C == self.channels
        assert (D, H, W) == self.grid_size

        # --- voxel-level branch ---
        x_32 = self.iwe_32(x)
        F32 = self._apply_mamba_voxel(
            x_32, self.seq_order_32, self.ln_32, self.mamba_32
        )

        # --- chunk_size = R_a branch ---
        if return_memory_attn and self.use_memory_bank:
            F_a, attn_a = self._apply_mamba_chunks(
                x, self.R_a, self.N_a, self.num_chunks_a,
                self.embed_a, self.unembed_a,
                self.ln_a, self.mamba_a,
                self.chunk_order_a,
                memory_bank=self.memory_bank_a,
                return_attn=True,
            )
        else:
            F_a = self._apply_mamba_chunks(
                x, self.R_a, self.N_a, self.num_chunks_a,
                self.embed_a, self.unembed_a,
                self.ln_a, self.mamba_a,
                self.chunk_order_a,
                memory_bank=self.memory_bank_a,
                return_attn=False,
            )
            attn_a = None

        # --- chunk_size = R_b branch ---
        if return_memory_attn and self.use_memory_bank:
            F_b, attn_b = self._apply_mamba_chunks(
                x, self.R_b, self.N_b, self.num_chunks_b,
                self.embed_b, self.unembed_b,
                self.ln_b, self.mamba_b,
                self.chunk_order_b,
                memory_bank=self.memory_bank_b,
                return_attn=True,
            )
        else:
            F_b = self._apply_mamba_chunks(
                x, self.R_b, self.N_b, self.num_chunks_b,
                self.embed_b, self.unembed_b,
                self.ln_b, self.mamba_b,
                self.chunk_order_b,
                memory_bank=self.memory_bank_b,
                return_attn=False,
            )
            attn_b = None

        # --- multi-scale fusion ---
        Fe = F32 + F_a + F_b + x

        if return_memory_attn:
            aux = {
                "memory_attn_a": attn_a,
                "memory_attn_b": attn_b,
            }
            return Fe, aux

        return Fe

class DecoderDualVoxelMamba3D(nn.Module):
    def __init__(
        self,
        channels: int,
        grid_size=(16, 16, 16),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_iwe: bool = False,
        iwe_window_size=(4, 4, 4),
    ):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.use_iwe = use_iwe

        D, H, W = grid_size

        # --- OPTIONAL IWE ---
        if self.use_iwe:
            self.iwe_fwd = ImplicitWindowEmbedding3D(
                channels=channels,
                grid_size=grid_size,
                window_size=iwe_window_size,
            )
            self.iwe_bwd = ImplicitWindowEmbedding3D(
                channels=channels,
                grid_size=grid_size,
                window_size=iwe_window_size,
            )

        # --- forward branch ---
        self.ln_fwd = nn.LayerNorm(channels)
        self.mamba_fwd = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # --- backward branch ---
        self.ln_bwd = nn.LayerNorm(channels)
        self.mamba_bwd = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Hilbert order
        order = hilbert_order_3d(D, H, W)
        self.register_buffer("seq_order", order)

        rev_order = torch.flip(order, dims=[0])
        self.register_buffer("seq_order_rev", rev_order)

        inv = torch.argsort(order)
        inv_rev = torch.argsort(rev_order)
        self.register_buffer("inv_order", inv)
        self.register_buffer("inv_order_rev", inv_rev)

    def _apply_mamba_seq(self, x, seq_order, inv_order, ln_layer, mamba_module):
        B, C, D, H, W = x.shape
        N = D * H * W

        x_flat = x.view(B, C, N).transpose(1, 2)
        x_flat = x_flat[:, seq_order, :]

        x_flat = ln_layer(x_flat)
        y_flat = mamba_module(x_flat)

        y_flat = y_flat[:, inv_order, :]
        y = y_flat.transpose(1, 2).view(B, C, D, H, W)
        return y

    def forward(self, f: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = f.shape
        assert d.shape == f.shape

        # --- APPLY IWE IF ENABLED ---
        if self.use_iwe:
            f = self.iwe_fwd(f)
            d = self.iwe_bwd(d)

        # forward branch (DINO side)
        Ff = self._apply_mamba_seq(
            f, self.seq_order, self.inv_order, self.ln_fwd, self.mamba_fwd
        )

        # backward branch (decoder side)
        Fb = self._apply_mamba_seq(
            d, self.seq_order, self.inv_order, self.ln_bwd, self.mamba_bwd
        )

        # fusion
        Fe = Ff + Fb + d
        return Fe
