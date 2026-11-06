# model.py

from typing import Tuple, Dict, Any

import torch
import torch.nn as nn

# model.py
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================
# TSDF PatchEmbed3D
# ------------------------------------------------------------
# VGGT reference:
#   vggt/layers/patch_embed.py :: class PatchEmbed
# does:
#   Conv2d(in_chans, embed_dim, kernel=patch, stride=patch)
#   -> flatten -> [B, N, embed_dim]
#
# Here (3D):
#   Conv3d(in_chans, embed_dim, kernel=patch, stride=patch)
#   -> flatten -> [B, N, embed_dim] where N=(32/patch)^3
# ============================================================
def make_3tuple(x: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if isinstance(x, tuple):
        assert len(x) == 3
        return x
    assert isinstance(x, int)
    return (x, x, x)


class PatchEmbed3D(nn.Module):
    """
    3D cube to patch embedding: (B,C,D,H,W) -> (B,N,embed_dim)
    For TSDF: x is [B,1,32,32,32]

    If patch_size=4:
      grid = 8x8x8, N=512.
    """
    def __init__(
        self,
        cube_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 1,
        embed_dim: int = 1024,
        flatten_embedding: bool = True,
    ):
        super().__init__()
        self.cube_size = make_3tuple(cube_size) if isinstance(cube_size, int) else cube_size
        self.patch_size = make_3tuple(patch_size) if isinstance(patch_size, int) else patch_size

        cd, ch, cw = self.cube_size
        pd, ph, pw = self.patch_size
        assert cd % pd == 0 and ch % ph == 0 and cw % pw == 0, "cube_size must be divisible by patch_size"

        self.grid_size = (cd // pd, ch // ph, cw // pw)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,32,32,32]
        x = self.proj(x)  # [B, embed_dim, Gd, Gh, Gw]
        if self.flatten_embedding:
            x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, embed_dim]
        return x


# ============================================================
# Attention + MLP + Block
# ------------------------------------------------------------
# VGGT reference:
#   vggt/layers/attention.py :: class Attention (qkv + SDPA)
#   vggt/layers/mlp.py       :: class Mlp
#   vggt/layers/block.py     :: NestedTensorBlock as Block (pre-norm)
#
# We implement the same essential mechanics (pre-norm, qkv, SDPA, MLP).
# ============================================================
class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        hidden = hidden_features if hidden_features is not None else 4 * in_features
        self.fc1 = nn.Linear(in_features, hidden, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Matches the shape contract of VGGT Attention:
      input  x: [B, N, C]
      output   : [B, N, C]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        fused_attn: bool = True,  # VGGT uses SDPA by default
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.attn_drop_p = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.fused_attn = fused_attn

        # VGGT supports optional qk_norm; we keep it for API parity
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: [B, heads, N, head_dim]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=(self.attn_drop_p if self.training else 0.0),
            )  # [B, heads, N, head_dim]
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop_p, training=self.training)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Pre-norm transformer block:
      x = x + Attn(LN(x))
      x = x + MLP(LN(x))

    VGGT reference:
      vggt/layers/block.py :: NestedTensorBlock forward does the same pattern.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
            bias=ffn_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FastDensify3D(nn.Module):
    def __init__(self, in_c=512, hidden_c=256, out_c=1024):
        super().__init__()
        self.reduce = nn.Conv3d(in_c, hidden_c, 1)

        self.refine8 = nn.Sequential(
            nn.Conv3d(hidden_c, hidden_c, 3, padding=1),
            nn.GELU(),
        )
        
        self.refine16 = nn.Sequential(
            nn.Conv3d(hidden_c, hidden_c, 3, padding=1),
            nn.GELU(),
        )
        self.refine32 = nn.Sequential(
            nn.Conv3d(hidden_c, hidden_c, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_c, hidden_c, 3, padding=1),
            nn.GELU(),
        )

        self.proj = nn.Conv3d(hidden_c, out_c, 1)

    def forward(self, x):
        x = self.reduce(x)  # [B,hidden,4,4,4]

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)  # 8
        x = self.refine8(x)

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)  # 16
        x = self.refine16(x)

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)  # 32
        x = self.refine32(x)

        x = self.proj(x)
        return x

# ============================================================
# DinoVisionTransformer3D (TSDF version of VGGT's DinoVisionTransformer)
# ------------------------------------------------------------
# VGGT reference:
#   vggt/layers/vision_transformer.py :: class DinoVisionTransformer
#
# Key methods we mirror:
#   - interpolate_pos_encoding (2D -> here 3D trilinear)
#   - prepare_tokens_with_masks
#   - forward_features (+ list version not necessary for TSDF)
#   - get_intermediate_layers (supports reshape=True, norm=True)
# ============================================================
class DinoVisionTransformer3D_TSDF(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        ma = dict(config.get("model_args", {}))  # model_args
        self._cfg = config                        # optional: keep full cfg for debugging

        cube_size = int(ma.get("cube_size", 32))
        patch_size = int(ma.get("patch_size", 8))
        in_chans = int(ma.get("in_chans", 1))
        embed_dim = int(ma.get("embed_dim", 1024))
        hidden_c = int(ma.get("hidden_c", 256))
        teacher_embed_dim = int(ma.get("teacher_embed_dim", 1024))
        depth = int(ma.get("depth", 6))
        num_heads = int(ma.get("num_heads", 16))
        mlp_ratio = float(ma.get("mlp_ratio", 4.0))
        qkv_bias = bool(ma.get("qkv_bias", True))
        ffn_bias = bool(ma.get("ffn_bias", True))
        proj_bias = bool(ma.get("proj_bias", True))
        qk_norm = bool(ma.get("qk_norm", False))
        num_register_tokens = int(ma.get("num_register_tokens", 0))
        use_reentrant = bool(ma.get("use_reentrant", False))

        # modules that are classes can also be configurable, but keep stable for now:
        embed_layer = PatchEmbed3D
        block_fn = Block

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.use_reentrant = use_reentrant

        self.patch_embed = embed_layer(
            cube_size=cube_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=True,
        )
        num_patches = self.patch_embed.num_patches
        gd, gh, gw = self.patch_embed.grid_size  # e.g., (8,8,8)

        # VGGT reference: self.cls_token, self.mask_token, self.pos_embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.register_tokens = None
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))

        self.mask_token.requires_grad_(False)
        if self.register_tokens is not None:
            self.register_tokens.requires_grad_(False)

        # Learned absolute position embedding over (cls + patches) like VGGT
        # VGGT stores pos_embed as [1, 1+num_patches, embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        
        self.densify = bool(ma.get("densify", False))
        if self.densify:
            self.to_dense = FastDensify3D(in_c=embed_dim, hidden_c=hidden_c, out_c=teacher_embed_dim)


        # Transformer blocks (same loop structure as VGGT)
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                ffn_bias=ffn_bias,
                proj_bias=proj_bias,
                drop=0.0,
                attn_drop=0.0,
                qk_norm=qk_norm,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),  # VGGT uses eps=1e-6
                fused_attn=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self._grid_size = (gd, gh, gw)
        self._patch_size = patch_size
        self._cube_size = cube_size
        self._num_patches = num_patches

    # ------------------------------------------------------------
    # VGGT reference:
    #   DinoVisionTransformer.interpolate_pos_encoding(self, x, w, h)
    #
    # Here we do 3D:
    #   interpolate pos embedding from (Gd0,Gh0,Gw0) -> (Gd,Gh,Gw)
    # using trilinear.
    # ------------------------------------------------------------
    def interpolate_pos_encoding(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        # x: [B, 1+N, C] (or includes register tokens later)
        npatch = x.shape[1] - 1  # exclude CLS; register tokens are inserted after pos in prepare_tokens_with_masks (like VGGT)
        N0 = self._num_patches
        if npatch == N0 and (D, H, W) == (self._cube_size, self._cube_size, self._cube_size):
            return self.pos_embed

        # pos_embed: [1, 1+N0, C] -> split cls and patch
        class_pos = self.pos_embed[:, :1]                    # [1,1,C]
        patch_pos = self.pos_embed[:, 1:]                    # [1,N0,C]

        gd0, gh0, gw0 = self._grid_size
        C = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, gd0, gh0, gw0, C).permute(0, 4, 1, 2, 3)  # [1,C,gd0,gh0,gw0]

        gd = D // self._patch_size
        gh = H // self._patch_size
        gw = W // self._patch_size

        patch_pos = F.interpolate(patch_pos, size=(gd, gh, gw), mode="trilinear", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 4, 1).reshape(1, -1, C)  # [1, N, C]
        return torch.cat([class_pos, patch_pos], dim=1).to(x.dtype)

    # ------------------------------------------------------------
    # VGGT reference:
    #   DinoVisionTransformer.prepare_tokens_with_masks(self, x, masks)
    # ------------------------------------------------------------
    def prepare_tokens_with_masks(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,1,32,32,32]
        B, nc, D, H, W = x.shape
        x = self.patch_embed(x)  # [B,N,C]

        if masks is not None:
            # masks expected [B,N] boolean
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        # prepend CLS
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)  # [B,1+N,C]
        x = x + self.interpolate_pos_encoding(x, D, H, W)            # add pos

        # insert register tokens after cls (same as VGGT)
        if self.register_tokens is not None:
            x = torch.cat(
                [x[:, :1], self.register_tokens.expand(B, -1, -1), x[:, 1:]],
                dim=1,
            )
        return x

    # ------------------------------------------------------------
    # VGGT reference:
    #   DinoVisionTransformer.forward_features(self, x, masks=None)
    # ------------------------------------------------------------
    def forward_features(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        DINO-like forward_features:
        returns normalized cls, reg, patch tokens + prenorm tokens
        """
        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            if self.training:
                x = checkpoint(blk, x, use_reentrant=self.use_reentrant)
            else:
                x = blk(x)

        x_norm = self.norm(x)

        # layout: [CLS][REG...][PATCH...]
        cls = x_norm[:, 0]  # [B,C]
        reg = x_norm[:, 1 : 1 + self.num_register_tokens] if self.num_register_tokens > 0 else x_norm[:, 0:0]  # [B,R,C]
        patch = x_norm[:, 1 + self.num_register_tokens :]  # [B,N,C]

        return {
            "x_norm_clstoken": cls,
            "x_norm_regtokens": reg,
            "x_norm_patchtokens": patch,
            "x_prenorm": x,
            "masks": masks,
        }

    # ------------------------------------------------------------
    # VGGT reference:
    #   DinoVisionTransformer.get_intermediate_layers(..., reshape=True, norm=True)
    #
    # This is what you use in your DINO extraction:
    #   feats = model.get_intermediate_layers(img, n=range(n_layers), reshape=True, norm=True)
    # -> feats[-1] = [B, embed_dim, Hp, Wp] in 2D
    #
    # Here:
    # -> feats[-1] = [B, embed_dim, Gd, Gh, Gw] in 3D
    # ------------------------------------------------------------
    def get_intermediate_layers(
        self,
        x: torch.Tensor,                          # [B,1,32,32,32]
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        norm: bool = False,
        return_class_token: bool = False,
    ) -> List[torch.Tensor]:
        # prepare tokens (no masks for intermediate extraction)
        B, _, D, H, W = x.shape
        x = self.prepare_tokens_with_masks(x, masks=None)

        total_blocks = len(self.blocks)
        if isinstance(n, int):
            blocks_to_take = list(range(total_blocks - n, total_blocks))
        else:
            blocks_to_take = list(n)

        outs: List[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                y = self.norm(x) if norm else x
                if return_class_token:
                    cls = y[:, 0]
                # strip prefix tokens (cls + registers)
                y = y[:, self.num_register_tokens + 1 :]  # [B, N, C]
                if reshape:
                    gd, gh, gw = (D // self._patch_size, H // self._patch_size, W // self._patch_size)
                    y = y.transpose(1, 2).contiguous().view(B, self.embed_dim, gd, gh, gw)  # [B,C,Gd,Gh,Gw]
                if return_class_token:
                    outs.append((y, cls))
                else:
                    outs.append(y)

        return outs

    # ------------------------------------------------------------
    # Standard forward for training
    # ------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor], mask=None) -> Dict[str, torch.Tensor]:
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        out = self.forward_features(x, masks=None)
        patch = out["x_norm_patchtokens"]  # [B,N,C]

        # reshape patches to grid
        B, _, D, H, W = x.shape
        gd, gh, gw = (D // self._patch_size, H // self._patch_size, W // self._patch_size)
        patch_grid = patch.transpose(1, 2).contiguous().view(B, self.embed_dim, gd, gh, gw)  # [B,C,gd,gh,gw]

        if self.densify:
            dense = self.to_dense(patch_grid)  # [B,C,32,32,32]
        else:
            dense = patch_grid

        return {
            "dino_proj": dense,                    # voxel feature target
            "x_norm_clstoken": out["x_norm_clstoken"],
            "x_norm_regtokens": out["x_norm_regtokens"],
            "x_norm_patchtokens": out["x_norm_patchtokens"],
        }



# ============================================================
# TSDF equivalent of VGGT's vit_large
# ------------------------------------------------------------
# VGGT reference (your snippet):
#   def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
#       model = DinoVisionTransformer(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, ...)
# ============================================================
def vit_large_tsdf(patch_size: int = 4, num_register_tokens: int = 0, **kwargs) -> DinoVisionTransformer3D_TSDF:
    model = DinoVisionTransformer3D_TSDF(
        cube_size=32,
        patch_size=patch_size,
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block),  # VGGT passes partial(Block, attn_class=MemEffAttention); we use Attention inside Block
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model



class ConvBlock3D(nn.Module):
    """
    Conv3d -> (BN) -> SiLU -> Conv3d -> (BN) -> SiLU
    """
    def __init__(self, in_c: int, out_c: int, use_bn: bool = False, dropout: float = 0.0):
        super().__init__()
        self.use_bn = use_bn

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.act = nn.SiLU(inplace=True)

        self.drop = nn.Dropout3d(dropout) if dropout > 0 else None

        if use_bn:
            self.bn1 = nn.BatchNorm3d(out_c)
            self.bn2 = nn.BatchNorm3d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.use_bn: x = self.bn1(x)
        x = self.act(x)
        if self.drop: x = self.drop(x)

        x = self.conv2(x)
        if self.use_bn: x = self.bn2(x)
        x = self.act(x)
        return x


class Down3D(nn.Module):
    """
    Downsample by 2, then conv block
    """
    def __init__(self, in_c: int, out_c: int, use_bn: bool = False, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block = ConvBlock3D(in_c, out_c, use_bn=use_bn, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.block(x)
        return x


class Up3D(nn.Module):
    """
    Upsample by 2 (trilinear), concat skip, then conv block.
    """
    def __init__(self, in_c: int, skip_c: int, out_c: int, use_bn: bool = False, dropout: float = 0.0):
        super().__init__()
        self.block = ConvBlock3D(in_c + skip_c, out_c, use_bn=use_bn, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: [B,C, d,h,w] -> up to match skip
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class DinoTSDFUNetStudent(nn.Module):
    """
    3D U-Net student for voxel-DINO distillation.

    Input:  batch["input_tsdf"] = [B,1,32,32,32]
    Output: pred_dino = [B,dino_dim,32,32,32]

    Also returns "features" (the last decoder feature) if you want.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        dino_dim = int(config.get("dino_dim", 1024))
        base_c   = int(config.get("base_c", 48))          # 32^3 is small, 48/64 works well
        use_bn   = bool(config.get("use_bn", False))
        dropout  = float(config.get("dropout", 0.0))
        self.normalize_out = bool(config.get("normalize_out", False))  # good for cosine loss

        # Encoder
        self.enc0 = ConvBlock3D(1, base_c, use_bn=use_bn, dropout=dropout)         # 32^3
        self.enc1 = Down3D(base_c, base_c * 2, use_bn=use_bn, dropout=dropout)     # 16^3
        self.enc2 = Down3D(base_c * 2, base_c * 4, use_bn=use_bn, dropout=dropout) # 8^3
        self.enc3 = Down3D(base_c * 4, base_c * 8, use_bn=use_bn, dropout=dropout) # 4^3

        # Bottleneck (still 4^3)
        self.mid = ConvBlock3D(base_c * 8, base_c * 8, use_bn=use_bn, dropout=dropout)

        # Decoder
        self.up2 = Up3D(in_c=base_c * 8, skip_c=base_c * 4, out_c=base_c * 8, use_bn=use_bn, dropout=dropout) # 8^3
        self.up1 = Up3D(in_c=base_c * 8, skip_c=base_c * 2, out_c=base_c * 8, use_bn=use_bn, dropout=dropout) # 16^3
        self.up0 = Up3D(in_c=base_c * 8, skip_c=base_c,     out_c=base_c * 8,     use_bn=use_bn, dropout=dropout) # 32^3

        # Head: project to DINO dim per voxel
        self.head = nn.Conv3d(base_c * 8, dino_dim, kernel_size=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        # Encoder
        s0 = self.enc0(x)     # [B, base, 32,32,32]
        s1 = self.enc1(s0)    # [B, 2b, 16,16,16]
        s2 = self.enc2(s1)    # [B, 4b, 8,8,8]
        s3 = self.enc3(s2)    # [B, 8b, 4,4,4]

        # Bottleneck
        m = self.mid(s3)      # [B, 8b, 4,4,4]

        # Decoder
        d2 = self.up2(m, s2)  # [B, 4b, 8,8,8]
        d1 = self.up1(d2, s1) # [B, 2b, 16,16,16]
        d0 = self.up0(d1, s0) # [B, b,  32,32,32]

        pred = self.head(d0)  # [B, dino_dim, 32,32,32]

        if self.normalize_out:
            # L2 normalize over channel dim => makes cosine loss behave nicely
            pred = F.normalize(pred, p=2, dim=1, eps=1e-6)

        return {
            "dino_proj": pred,      # [B,dino_dim,32,32,32]
            "features": d0,         # [B,base_c,32,32,32]
        }

class DinoTSDFUNetStudent_classic(nn.Module):
    """
    Baseline-style encoder-decoder using BasicBlock_large only,
    with skip concatenations, producing voxel DINO features.

    Input : batch["input_tsdf"]     [B,1,32,32,32]
    Output: out["dino_proj"]        [B,dino_dim,32,32,32]
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        C = int(config.get("base_c", 64))
        dino_dim = int(config.get("dino_dim", 1024))

        use_batchnorm = bool(config.get("use_bn", True))
        dropout = float(config.get("dropout", 0.0))
        leaky = bool(config.get("leaky", False))
        self.normalize_out = bool(config.get("normalize_out", False))

        # ----------------------------
        # Encoder: 32 -> 16 -> 8 -> 4
        # ----------------------------
        # NOTE: Using k=4,s=2,p=1 is the classic "exact half" downsample.
        # If you insist on k=3 like your snippet, you can switch back.
        k_down = int(config.get("k_down", 4))  # default 4
        p_down = int(config.get("p_down", 1))

        self.enc1 = BasicBlock_large(
            'Encoder', 1, C // 2,
            kernel_size=k_down, stride=2, padding=p_down,
            use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout
        )  # [B,C/2,16,16,16]

        self.enc2 = BasicBlock_large(
            'Encoder', C // 2, C,
            kernel_size=k_down, stride=2, padding=p_down,
            use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout
        )  # [B,C,8,8,8]

        self.enc3 = BasicBlock_large(
            'Encoder', C, C * 2,
            kernel_size=k_down, stride=2, padding=p_down,
            use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout
        )  # [B,2C,4,4,4]

        

        # ----------------------------
        # Decoder: 4 -> 8 -> 16 -> 32
        # ----------------------------
        # Classic exact double upsample uses k=4,s=2,p=1,output_padding=0
        k_up = int(config.get("k_up", 4))       # default 4
        p_up = int(config.get("p_up", 1))
        op_up = int(config.get("op_up", 0))     # default 0 for k=4

        self.dec1 = BasicBlock_large(
            'Decoder', C * 2, C,
            kernel_size=k_up, stride=2, padding=p_up, output_padding=op_up,
            use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout
        )  # 4 -> 8, out [B,C,8,8,8]

        self.dec2 = BasicBlock_large(
            'Decoder', C + C, C,
            kernel_size=k_up, stride=2, padding=p_up, output_padding=op_up,
            use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout
        )  # 8 -> 16, input cat(f2,d1) => 2C, out [B,C/2,16,16,16]

        self.dec3 = BasicBlock_large(
            'Decoder', C + C // 2, C,
            kernel_size=k_up, stride=2, padding=p_up, output_padding=op_up,
            use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout
        )  # 16 -> 32, input cat(f1,d2) => C, out [B,C,32,32,32]

        # Head: project to DINO dim
        self.conv_last = nn.Conv3d(C, dino_dim, kernel_size=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        # Encoder
        f1 = self.enc1(x)        # [B,C/2,16,16,16]
        f2 = self.enc2(f1)       # [B,C,  8, 8, 8]
        f3 = self.enc3(f2)       # [B,2C, 4, 4, 4]

        # Decoder
        d1_in = f3               # [B,2C,4,4,4]
        d1 = self.dec1(d1_in)    # [B,C,8,8,8]

        d2_in = torch.cat([f2, d1], dim=1)  # [B,2C,8,8,8]
        d2 = self.dec2(d2_in)               # [B,C/2,16,16,16]

        d3_in = torch.cat([f1, d2], dim=1)  # [B,C,16,16,16]
        d3 = self.dec3(d3_in)               # [B,C,32,32,32]

        pred = self.conv_last(d3)           # [B,dino_dim,32,32,32]

        if self.normalize_out:
            pred = F.normalize(pred, p=2, dim=1, eps=1e-6)

        return {
            "dino_proj": pred,
            "features": d3,
        }


class DinoTSDFUNetStudent_classic_masked(nn.Module):
    """
    Your classic BasicBlock_large encoder-decoder + a learned voxel mask.

    Output is hard-gated (optionally straight-through during training):
      pred = pred_raw * mask

    Returns:
      dino_proj      [B,C,32,32,32] gated
      dino_proj_raw  [B,C,32,32,32] ungated (useful for debugging)
      mask           [B,1,32,32,32] soft mask in [0,1]
      mask_logits    [B,1,32,32,32]
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        C = int(config.get("base_c", 64))
        dino_dim = int(config.get("dino_dim", 1024))

        use_batchnorm = bool(config.get("use_bn", True))
        dropout = float(config.get("dropout", 0.0))
        leaky = bool(config.get("leaky", False))

        # IMPORTANT: to allow exact zeros after gating, keep normalize_out False.
        self.normalize_out = bool(config.get("normalize_out", False))

        # mask controls
        self.hard_mask_train = bool(config.get("hard_mask_train", False))
        self.hard_mask_thresh = float(config.get("hard_mask_thresh", 0.5))
        self.mask_temperature = float(config.get("mask_temperature", 1.0))  # <1 => sharper sigmoid

        # ----------------------------
        # Encoder: 32 -> 16 -> 8 -> 4
        # ----------------------------
        k_down = int(config.get("k_down", 4))
        p_down = int(config.get("p_down", 1))

        self.enc1 = BasicBlock_large('Encoder', 1, C // 2,
                                     kernel_size=k_down, stride=2, padding=p_down,
                                     use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout)
        self.enc2 = BasicBlock_large('Encoder', C // 2, C,
                                     kernel_size=k_down, stride=2, padding=p_down,
                                     use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout)
        self.enc3 = BasicBlock_large('Encoder', C, C * 2,
                                     kernel_size=k_down, stride=2, padding=p_down,
                                     use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout)

        # ----------------------------
        # Decoder: 4 -> 8 -> 16 -> 32
        # ----------------------------
        k_up = int(config.get("k_up", 4))
        p_up = int(config.get("p_up", 1))
        op_up = int(config.get("op_up", 0))

        self.dec1 = BasicBlock_large('Decoder', C * 2, C,
                                     kernel_size=k_up, stride=2, padding=p_up, output_padding=op_up,
                                     use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout)
        self.dec2 = BasicBlock_large('Decoder', C + C, C,
                                     kernel_size=k_up, stride=2, padding=p_up, output_padding=op_up,
                                     use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout)
        self.dec3 = BasicBlock_large('Decoder', C + C // 2, C,
                                     kernel_size=k_up, stride=2, padding=p_up, output_padding=op_up,
                                     use_batchnorm=use_batchnorm, leaky=leaky, dropout=dropout)

        # Head: project to DINO dim
        self.conv_last = nn.Conv3d(C, dino_dim, kernel_size=1)

        # NEW: mask head (predict 1-channel voxel mask from final decoder features)
        # Keep it light; 3x3 + 1x1 is enough.
        self.mask_head = nn.Sequential(
            nn.Conv3d(C, max(C // 2, 16), kernel_size=3, padding=1),
            nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(max(C // 2, 16), 1, kernel_size=1),
        )

        # self.cls_from = str(config.get("cls_from", "enc3"))  # "enc2" or "enc3"
        # cls_out = (C * 4)  # enc3 channels

        # self.cls_conv = nn.Sequential(
        #     nn.Conv3d(C * 2, cls_out, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv3d(cls_out, cls_out, kernel_size=2, stride=2),  # 4→2
        #     nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv3d(cls_out, cls_out, kernel_size=2, stride=2),  # 2→1
        # )

        # self.cls_head = nn.Sequential(
        #     nn.Flatten(),                 # [B,cls_in,1,1,1] -> [B,cls_in]
        #     nn.Linear(cls_out, dino_dim),  # -> [B,1024]
        # )
        # self.norm = nn.LayerNorm(dino_dim, eps=1e-6)

    def _st_hard_mask(self, m_soft: torch.Tensor) -> torch.Tensor:
        """Straight-through hard mask: forward uses hard 0/1, backward uses soft grads."""
        m_hard = (m_soft > self.hard_mask_thresh).float()
        return m_hard.detach() - m_soft.detach() + m_soft

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        # Encoder
        f1 = self.enc1(x)        # [B,C/2,16,16,16]
        f2 = self.enc2(f1)       # [B,C,  8, 8, 8]
        f3 = self.enc3(f2)       # [B,2C, 4, 4, 4]

        #cls token
        # cls_feat = self.cls_conv(f3)   # [B,2C,1,1,1]   
        # dino_cls = self.cls_head(cls_feat)  # [B,1024]
        # dino_cls = self.norm(dino_cls)  # taken from dino 

        # Decoder
        d1 = self.dec1(f3)                       # [B,C,8,8,8]
        d2 = self.dec2(torch.cat([f2, d1], 1))   # [B,C,16,16,16]
        d3 = self.dec3(torch.cat([f1, d2], 1))   # [B,C,32,32,32]

        pred_raw = self.conv_last(d3)            # [B,dino_dim,32,32,32]

        # mask
        mask_logits = self.mask_head(d3)         # [B,1,32,32,32]
        mask_soft = torch.sigmoid(mask_logits / max(self.mask_temperature, 1e-6))

        if self.training and self.hard_mask_train:
            mask_used = self._st_hard_mask(mask_soft)
        else:
            mask_used = mask_soft

        pred = pred_raw * mask_used              # HARD capability to be exactly 0 where mask=0

        # If you want exact zeros, do NOT turn on normalize_out.
        if self.normalize_out:
            pred = F.normalize(pred, p=2, dim=1, eps=1e-6)

        return {
            "dino_proj": pred,
            "dino_proj_raw": pred_raw,
            "mask": mask_soft,
            "mask_logits": mask_logits,
            "features": d3,
            #"x_norm_clstoken": dino_cls,
        }


# -------------------------
# ViT building blocks
# -------------------------

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-LN ViT block (same structure as DINO ViT):
      x = x + MHA(LN(x))
      x = x + MLP(LN(x))
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.proj_drop(attn_out)

        h = self.norm2(x)
        x = x + self.mlp(h)
        return x





# -------------------------
# 3D ViT "DINO-like" predictor
# -------------------------

class DinoLikeViT3D_TSDF(nn.Module):
    """
    DINOv3-ViT-like (architecture) in 3D for TSDF.

    Patchify:
      [B,1,32,32,32] --(patch=4)--> N=512 tokens

    Transformer (ViT-L-like defaults):
      embed_dim=1024, depth=24, heads=16, mlp_ratio=4

    Outputs:
      feat_512 : [B,  512,32,32,32]
      dino_proj: [B, 1024,32,32,32]  (for your fused DINO supervision)
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        patch_size = int(config.get("patch_size", 4))      # <-- your choice
        embed_dim  = int(config.get("embed_dim", 1024))    # ViT-L-like
        depth      = int(config.get("depth", 24))          # ViT-L-like
        n_heads    = int(config.get("n_heads", 16))        # ViT-L-like
        mlp_ratio  = float(config.get("mlp_ratio", 4.0))   # ViT-L-like

        attn_dropout = float(config.get("attn_dropout", 0.0))
        proj_dropout = float(config.get("proj_dropout", 0.0))
        mlp_dropout  = float(config.get("mlp_dropout", 0.0))

        self.out_feat_dim  = int(config.get("out_feat_dim", 512))   # dense feature channels
        self.dino_dim      = int(config.get("dino_dim", 1024))      # supervision channels
        self.normalize_out = bool(config.get("normalize_out", False))
        self.use_cls_token = bool(config.get("use_cls_token", False))  # DINO has CLS; optional

        # validate
        if 32 % patch_size != 0:
            raise ValueError(f"patch_size must divide 32. Got {patch_size}")

        Dp = Hp = Wp = 32 // patch_size
        num_tokens = Dp * Hp * Wp  # for patch tokens

        # patch embed
        self.patch = PatchEmbed3D(in_ch=1, embed_dim=embed_dim, patch_size=patch_size)

        # CLS token + pos embed (DINO-style)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.use_cls_token else None
        total_tokens = num_tokens + (1 if self.use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                n_heads=n_heads,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                mlp_ratio=mlp_ratio,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # token -> dense (unpatchify)
        # tokens -> [B,E,D',H',W'] then upsample back to 32^3
        self.unpatch = nn.ConvTranspose3d(
            in_channels=embed_dim,
            out_channels=self.out_feat_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            output_padding=0,
        )

        # small refinement at full res (optional but helps)
        self.refine = nn.Sequential(
            nn.Conv3d(self.out_feat_dim, self.out_feat_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(self.out_feat_dim, self.out_feat_dim, kernel_size=3, padding=1),
        )

        # final projection to DINO-dim target
        self.to_dino = nn.Conv3d(self.out_feat_dim, self.dino_dim, kernel_size=1)

        self._patch_size = patch_size

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        # patchify -> tokens
        tok, (Dp, Hp, Wp) = self.patch(x)  # [B,N,E], N=Dp*Hp*Wp

        # add CLS token (optional)
        if self.use_cls_token:
            B = tok.shape[0]
            cls = self.cls_token.expand(B, -1, -1)           # [B,1,E]
            tok = torch.cat([cls, tok], dim=1)               # [B,1+N,E]

        # add pos embedding
        tok = tok + self.pos_embed

        # transformer encoder
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)  # [B,1+N,E] or [B,N,E]

        # remove CLS for dense reconstruction
        if self.use_cls_token:
            tok_p = tok[:, 1:, :]  # [B,N,E]
            cls_out = tok[:, :1, :] # [B,1,E] (if you want it)
        else:
            tok_p = tok
            cls_out = None

        # tokens -> coarse grid [B,E,D',H',W']
        B, N, E = tok_p.shape
        grid = tok_p.transpose(1, 2).contiguous().view(B, E, Dp, Hp, Wp)

        # unpatchify to dense 32^3 with 512 channels
        feat_512 = self.unpatch(grid)              # [B,512,32,32,32]
        feat_512 = feat_512 + self.refine(feat_512)

        # project to 1024 to match fused DINO
        dino_proj = self.to_dino(feat_512)         # [B,1024,32,32,32]
        if self.normalize_out:
            dino_proj = F.normalize(dino_proj, p=2, dim=1, eps=1e-6)

        out = {
            "feat_512": feat_512,
            "dino_proj": dino_proj,
            "tokens": tok_p,   # patch tokens only [B,N,E]
        }
        if cls_out is not None:
            out["cls_token"] = cls_out
        return out



class ResBlock3D(nn.Module):
    def __init__(self, c, use_bn=False, dropout=0.0):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv3d(c, c, 3, padding=1)
        self.conv2 = nn.Conv3d(c, c, 3, padding=1)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else None
        if use_bn:
            self.bn1 = nn.BatchNorm3d(c)
            self.bn2 = nn.BatchNorm3d(c)

    def forward(self, x):
        y = self.conv1(x)
        if self.use_bn: y = self.bn1(y)
        y = self.act(y)
        if self.drop: y = self.drop(y)
        y = self.conv2(y)
        if self.use_bn: y = self.bn2(y)
        return self.act(x + y)


class DinoTSDFStudent(nn.Module):
    """
    Ideal simple student: TSDF -> voxel-DINO features at 32^3.

    Input:  x [B,1,32,32,32]
    Output: y [B,dino_dim,32,32,32]
    """
    def __init__(
        self,
        dino_dim: int = 1024,   # ViT-L is usually 1024 final dim
        base_c: int = 96,
        num_blocks: int = 6,
        use_bn: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(1, base_c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_c, base_c, 3, padding=1),
            nn.SiLU(inplace=True),
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock3D(base_c, use_bn=use_bn, dropout=dropout))
        self.trunk = nn.Sequential(*blocks)

        # small refinement head
        self.head = nn.Sequential(
            nn.Conv3d(base_c, base_c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_c, dino_dim, 1),
        )

    def forward(self, input_tsdf: torch.Tensor) -> torch.Tensor:
        x = self.stem(input_tsdf)
        x = self.trunk(x)
        y = self.head(x)
        return y


class BasicBlock_large(nn.Module):
    def __init__(self,
                 mode: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 1,
                 output_padding: int = 1,
                 use_batchnorm: bool = True,
                 leaky: bool = False,
                 dropout: float = 0.0):
        super(BasicBlock_large, self).__init__()

        if mode == 'Encoder':
            self._conv1 = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        elif mode == 'Decoder':
            self._conv1 = nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            )
        else:
            raise ValueError("Wrong mode, please enter 'Encoder' or 'Decoder'.")

        self._conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self._conv3 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self._dropout = nn.Dropout3d(p=dropout) if dropout > 0 else None

        self._relu = nn.LeakyReLU(0.2) if leaky else nn.ReLU(inplace=True)
        self._use_batchnorm = use_batchnorm
        if self._use_batchnorm:
            self._bn_1 = nn.BatchNorm3d(out_channels)
            self._bn_2 = nn.BatchNorm3d(out_channels)
            self._bn_3 = nn.BatchNorm3d(out_channels)
        # else:
        #     # self._conv1 = nn.utils.weight_norm(self._conv1, name='weight')
        #     # self._conv2 = nn.utils.weight_norm(self._conv2, name='weight')
        #     # self._conv3 = nn.utils.weight_norm(self._conv3, name='weight')

        #     self._conv1 = nn.utils.parametrizations.weight_norm(self._conv1, name='weight')
        #     self._conv2 =  nn.utils.parametrizations.weight_norm(self._conv2, name='weight')
        #     self._conv3 =  nn.utils.parametrizations.weight_norm(self._conv3, name='weight')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_batchnorm:
            out = self._conv1(x)
            out = self._bn_1(out)
            out = self._relu(out)

            identity = out
            out = self._conv2(out)
            out = self._bn_2(out)
            out = self._relu(out)

            if self._dropout:
                out = self._dropout(out)

            out = self._conv3(out)
            out = self._bn_3(out)
        else:
            out = self._conv1(x)
            out = self._relu(out)

            identity = out
            out = self._conv2(out)
            out = self._relu(out)

            if self._dropout:
                out = self._dropout(out)

            out = self._conv3(out)

        out = out + identity
        out = self._relu(out)
        return out

class Baseline(nn.Module):
    """
    Simple encoder–decoder TSDF completion network 

    Config dict expected keys (with defaults):
      - channel_num   (int)
      - use_batchnorm (bool)
      - dropout       (float)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        C = config.get("channel_num", 64)
        use_batchnorm = config.get("use_batchnorm", True)
        dropout = config.get("dropout", 0.0)

        # Encoder: 32^3 -> 16^3 -> 8^3 -> 4^3
        self.enc1 = BasicBlock_large(
            'Encoder', 1, C // 2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C/2,16,16,16]

        self.enc2 = BasicBlock_large(
            'Encoder', C // 2, C,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C,8,8,8]

        self.enc3 = BasicBlock_large(
            'Encoder', C, C*2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C,4,4,4]

        

        # Decoder: 4^3 -> 8^3 -> 16^3 -> 32^3
        self.dec1 = BasicBlock_large(
            'Decoder', C*2, C,
            kernel_size=3, stride=2, padding=1, output_padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C,4,4,4] -> [B,C,8,8,8]

        self.dec2 = BasicBlock_large(
            'Decoder', C*2, C,
            kernel_size=3, stride=2, padding=1, output_padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C,8,8,8] -> [B,C/2,16,16,16]

        self.dec3 = BasicBlock_large(
            'Decoder', C + C // 2, C,
            kernel_size=3, stride=2, padding=1, output_padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C/2,16,16,16] -> [B,C/2,32,32,32]

        self.conv_last = nn.Conv3d(C, 1, 3, 1, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with key "input_tsdf": [B,1,32,32,32]

        Returns:
            {
                "pred_tsdf": [B,1,32,32,32],
                "latent":    [B,C],
                "dino_proj": [B,dino_dim],
            }
        """
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        # Encoder
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)  # [B,C,4,4,4]


        #decoder for 4
        final_shape = f3 #TODO train decoders as well, log testte sadece hypbonet train etmişsin encoder_transformerı train etmemişsin
        d1 = self.dec1(final_shape)

        #decoder for 8
        final_shape = torch.cat((f2, d1), dim=1)
        d2 = self.dec2(final_shape)

        #decoder for 16
        final_shape = torch.cat((f1, d2), dim=1)
        d3 = self.dec3(final_shape)

        pred_tsdf = self.conv_last(d3)  # [B,1,32,32,32]

        return {
            "pred_tsdf": pred_tsdf,
        }

class DinoTSDFNet(nn.Module):
    """
    Simple encoder–decoder TSDF completion network with latent code
    used for DINO distillation.

    Config dict expected keys (with defaults):
      - channel_num   (int)
      - use_batchnorm (bool)
      - dino_dim      (int)
      - dropout       (float)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        C = config.get("channel_num", 64)
        use_batchnorm = config.get("use_batchnorm", True)
        dino_dim = config.get("dino_dim", 768)
        dropout = config.get("dropout", 0.0)

        # Encoder: 32^3 -> 16^3 -> 8^3 -> 4^3
        self.enc1 = BasicBlock_large(
            'Encoder', 1, C // 2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C/2,16,16,16]

        self.enc2 = BasicBlock_large(
            'Encoder', C // 2, C,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C,8,8,8]

        self.enc3 = BasicBlock_large(
            'Encoder', C, C*2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C,4,4,4]

        # Latent
        self.latent_gap = nn.AdaptiveAvgPool3d(1)  # -> [B,C,1,1,1]
        self.latent_fc = nn.Identity()             # hook for extra MLP if needed

        self.proj_head = nn.Sequential(
            nn.Linear(C*2, C*2),
            nn.ReLU(inplace=True),
            nn.Linear(C*2, dino_dim),
        )

        # Decoder: 4^3 -> 8^3 -> 16^3 -> 32^3
        self.dec1 = BasicBlock_large(
            'Decoder', C*2, C,
            kernel_size=3, stride=2, padding=1, output_padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C,4,4,4] -> [B,C,8,8,8]

        self.dec2 = BasicBlock_large(
            'Decoder', C*2, C,
            kernel_size=3, stride=2, padding=1, output_padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C,8,8,8] -> [B,C/2,16,16,16]

        self.dec3 = BasicBlock_large(
            'Decoder', C + C // 2, C,
            kernel_size=3, stride=2, padding=1, output_padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C/2,16,16,16] -> [B,C/2,32,32,32]

        self.conv_last = nn.Conv3d(C, 1, 3, 1, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with key "input_tsdf": [B,1,32,32,32]

        Returns:
            {
                "pred_tsdf": [B,1,32,32,32],
                "latent":    [B,C],
                "dino_proj": [B,dino_dim],
            }
        """
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        # Encoder
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)  # [B,C,4,4,4]

        # Latent
        latent = self.latent_gap(f3).view(f3.size(0), -1)  # [B,C]
        latent = self.latent_fc(latent)
        dino_proj = self.proj_head(latent)  # [B,dino_dim]

        #decoder for 4
        final_shape = f3 #TODO train decoders as well, log testte sadece hypbonet train etmişsin encoder_transformerı train etmemişsin
        d1 = self.dec1(final_shape)

        #decoder for 8
        final_shape = torch.cat((f2, d1), dim=1)
        d2 = self.dec2(final_shape)

        #decoder for 16
        final_shape = torch.cat((f1, d2), dim=1)
        d3 = self.dec3(final_shape)

        pred_tsdf = self.conv_last(d3)  # [B,1,32,32,32]

        return {
            "pred_tsdf": pred_tsdf,
            "latent": latent,
            "dino_proj": dino_proj,
        }

class DinoTSDFEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        C         = config.get("channel_num", 64)
        use_bn    = config.get("use_batchnorm", True)
        dino_dim  = config.get("dino_dim", 768)
        dropout   = config.get("dropout", 0.0)

        # Encoder: 32^3 -> 16^3 -> 8^3 -> 4^3
        self.enc1 = BasicBlock_large(
            'Encoder', 1, C // 2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_bn,
            dropout=dropout,
        )  # [B,C/2,16,16,16]

        self.enc2 = BasicBlock_large(
            'Encoder', C // 2, C,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_bn,
            dropout=dropout,
        )  # [B,C,8,8,8]

        self.enc3 = BasicBlock_large(
            'Encoder', C, C*2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_bn,
            dropout=dropout,
        )  # [B,C,4,4,4]

        # Global latent (option A: GAP + MLP)
        self.latent_gap = nn.AdaptiveAvgPool3d(1)  # -> [B,C,1,1,1]
        self.latent_fc  = nn.Identity()

        self.proj_head = nn.Sequential(
            nn.Linear(C*2, C*2),
            nn.ReLU(inplace=True),
            nn.Linear(C*2, dino_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)      # [B,C,4,4,4]

        latent = self.latent_gap(f3).view(f3.size(0), -1)  # [B,C]
        latent = self.latent_fc(latent)
        dino_proj = self.proj_head(latent)                 # [B,dino_dim]

        return {
            "latent": latent,
            "dino_proj": dino_proj,
            "feat_4": f3,   # keep if you need it for something else
        }

class DinoTSDFEncoder4(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        C         = config.get("channel_num", 64)
        use_bn    = config.get("use_batchnorm", True)
        dino_dim  = config.get("dino_dim", 768)
        dropout   = config.get("dropout", 0.0)

        # Encoder: 32^3 -> 16^3 -> 8^3 -> 4^3
        self.enc1 = BasicBlock_large(
            'Encoder', 1, C // 2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_bn,
            dropout=dropout,
        )  # [B,C/2,16,16,16]

        self.enc2 = BasicBlock_large(
            'Encoder', C // 2, C,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_bn,
            dropout=dropout,
        )  # [B,C,8,8,8]

        self.enc3 = BasicBlock_large(
            'Encoder', C, C*2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_bn,
            dropout=dropout,
        )  # [B,2C,4,4,4]

        # 1x1x1 projection: 2C -> dino_dim at 4^3
        self.proj_conv = nn.Conv3d(
            in_channels=C*2,
            out_channels=dino_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        f1 = self.enc1(x)   # [B,C/2,16,16,16]
        f2 = self.enc2(f1)  # [B,C,8,8,8]
        f3 = self.enc3(f2)  # [B,2C,4,4,4]

        # Project to DINO-dim voxel features
        student = self.proj_conv(f3)  # [B,dino_dim,4,4,4]

        return {
            "features": f3,       # [B,2C,4,4,4] if you want it
            "dino_proj": student,    # [B,dino_dim,4,4,4] for distillation
        }

class DinoTSDFEncoder8(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        C         = config.get("channel_num", 64)
        use_bn    = config.get("use_batchnorm", True)
        dino_dim  = config.get("dino_dim", 768)
        dropout   = config.get("dropout", 0.0)

        # Encoder: 32^3 -> 16^3 -> 8^3 -> 4^3
        self.enc1 = BasicBlock_large(
            'Encoder', 1, C // 2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_bn,
            dropout=dropout,
        )  # [B,C/2,16,16,16]

        self.enc2 = BasicBlock_large(
            'Encoder', C // 2, C,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_bn,
            dropout=dropout,
        )  # [B,C,8,8,8]


        # 1x1x1 projection: 2C -> dino_dim at 4^3
        self.proj_conv = nn.Conv3d(
            in_channels=C,
            out_channels=dino_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        f1 = self.enc1(x)   # [B,C/2,16,16,16]
        f2 = self.enc2(f1)  # [B,C,8,8,8]

        # Project to DINO-dim voxel features
        student = self.proj_conv(f2)  # [B,dino_dim,4,4,4]

        return {
            "features": f2,       # [B,2C,4,4,4] if you want it
            "dino_proj": student,    # [B,dino_dim,4,4,4] for distillation
        }

class DinoTSDFEncoder16(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        C         = config.get("channel_num", 64)
        use_bn    = config.get("use_batchnorm", True)
        dino_dim  = config.get("dino_dim", 768)
        dropout   = config.get("dropout", 0.0)

        # Encoder: 32^3 -> 16^3 -> 8^3 -> 4^3
        self.enc1 = BasicBlock_large(
            'Encoder', 1, C // 2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_bn,
            dropout=dropout,
        )  # [B,C/2,16,16,16]



        # 1x1x1 projection: 2C -> dino_dim at 4^3
        self.proj_conv = nn.Conv3d(
            in_channels=C // 2,
            out_channels=dino_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        f1 = self.enc1(x)   # [B,C/2,16,16,16]

        # Project to DINO-dim voxel features
        student = self.proj_conv(f1)  # [B,dino_dim,4,4,4]

        return {
            "features": f1,       # [B,2C,4,4,4] if you want it
            "dino_proj": student,    # [B,dino_dim,4,4,4] for distillation
        }
    
class DinoTSDFEncoder32(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        C         = config.get("channel_num", 64)
        dino_dim  = config.get("dino_dim", 768)
        use_batchnorm = config.get("use_batchnorm", False)
        self.block1 = BasicBlock_large('Encoder', 1,  C//2, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm)
        self.block2 = BasicBlock_large('Encoder', C//2,  C, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm)
        self.to_dino = nn.Conv3d(C, dino_dim, kernel_size=1)

    def forward(self, batch):
        x = batch["input_tsdf"]  # [B,1,32,32,32]
        x = self.block1(x)      # [B,C,32,32,32]
        f = self.block2(x)      # [B,C,32,32,32]
        student = self.to_dino(f)  # [B,768,32,32,32]
        return {
                "features": f,       # [B,2C,4,4,4] if you want it
                "dino_proj": student,    # [B,dino_dim,4,4,4] for distillation
            }


class DinoFeatureUNet(nn.Module):
    """
    Encoder–decoder that takes an incomplete TSDF volume and predicts
    voxelized DINO features of the same spatial size (32^3).

    Config dict expected keys (with defaults):
      - channel_num   (int): base channels for UNet (e.g. 64 or 128)
      - use_batchnorm (bool)
      - dropout       (float)
      - dino_dim      (int): DINO feature dimension per voxel (e.g. 64, 128, 256...)
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        C            = config.get("channel_num", 64)
        use_batchnorm = config.get("use_batchnorm", True)
        dropout       = config.get("dropout", 0.0)
        self.dino_dim = config.get("dino_dim", 64)

        # --------- Encoder: 32^3 -> 16^3 -> 8^3 -> 4^3 ---------
        self.enc1 = BasicBlock_large(
            'Encoder', 1, C // 2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C/2,16,16,16]

        self.enc2 = BasicBlock_large(
            'Encoder', C // 2, C,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C,8,8,8]

        self.enc3 = BasicBlock_large(
            'Encoder', C, C * 2,
            kernel_size=3, stride=2, padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,2C,4,4,4]

        # # Optionally: a bottleneck conv to mix features at 4^3
        # self.bottleneck = nn.Sequential(
        #     nn.Conv3d(C * 2, C * 2, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(C * 2) if use_batchnorm else nn.Identity(),
        #     nn.ReLU(inplace=True),
        # )

        # --------- Decoder: 4^3 -> 8^3 -> 16^3 -> 32^3 ---------
        self.dec1 = BasicBlock_large(
            'Decoder', C * 2, C,
            kernel_size=3, stride=2, padding=1, output_padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,2C,4,4,4] -> [B,C,8,8,8]

        self.dec2 = BasicBlock_large(
            'Decoder', C * 2, C,
            kernel_size=3, stride=2, padding=1, output_padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,2C,8,8,8] -> [B,C,16,16,16]

        self.dec3 = BasicBlock_large(
            'Decoder', C + C // 2, C,
            kernel_size=3, stride=2, padding=1, output_padding=1,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )  # [B,C + C/2,16,16,16] -> [B,C,32,32,32]

        # Final projection: from UNet feature channels -> DINO channels
        # (3x3 keeps things local; 1x1 is also fine)
        self.conv_last = nn.Conv3d(C, self.dino_dim, kernel_size=3, stride=1, padding=1)

        # Optional global latent vector if you want a compact code
        #self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch:
              - "input_tsdf": [B,1,32,32,32]

        Returns:
            {
                "pred_dino": [B,dino_dim,32,32,32],
                "latent":    [B,2C],        # bottleneck-pooled
            }
        """
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        # ----- Encoder -----
        f1 = self.enc1(x)         # [B,C/2,16,16,16]
        f2 = self.enc2(f1)        # [B,C,8,8,8]
        f3 = self.enc3(f2)        # [B,2C,4,4,4]
        # f3 = self.bottleneck(f3)  # [B,2C,4,4,4] (optional)

        # compact latent (could be used for extra losses, etc.)
        #latent = self.global_pool(f3).flatten(1)  # [B,2C]

        # ----- Decoder -----
        d1 = self.dec1(f3)                 # [B,C,8,8,8]

        d1_cat = torch.cat((f2, d1), dim=1)  # [B,2C,8,8,8]
        d2 = self.dec2(d1_cat)              # [B,C,16,16,16]

        d2_cat = torch.cat((f1, d2), dim=1)  # [B,C + C/2,16,16,16]
        d3 = self.dec3(d2_cat)              # [B,C,32,32,32]

        pred_dino = self.conv_last(d3)      # [B,dino_dim,32,32,32]

        return {
            "dino_proj": pred_dino,
            #"latent":    latent,
        }


from typing import Dict, Any

class MaskUNet32(nn.Module):
    """
    3D U-Net mask predictor.
    Input : batch["input_tsdf"] -> [B,1,32,32,32]
    Output: logits/probs for missing + hallucinated masks.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        use_bn = config.get("use_batchnorm", False)
        base = config.get("mask_unet_base", 32)  # 16/32/48 etc.

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, 1, 1),
                nn.BatchNorm3d(cout) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, 3, 1, 1),
                nn.BatchNorm3d(cout) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(1, base)          # 32^3
        self.pool1 = nn.MaxPool3d(2)             # 16^3

        self.enc2 = conv_block(base, base*2)     # 16^3
        self.pool2 = nn.MaxPool3d(2)             # 8^3

        self.enc3 = conv_block(base*2, base*4)   # 8^3
        self.pool3 = nn.MaxPool3d(2)             # 4^3

        self.bottleneck = conv_block(base*4, base*8)  # 4^3

        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, 2)  # 4->8
        self.dec3 = conv_block(base*8, base*4)

        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, 2)  # 8->16
        self.dec2 = conv_block(base*4, base*2)

        self.up1 = nn.ConvTranspose3d(base*2, base, 2, 2)    # 16->32
        self.dec1 = conv_block(base*2, base)

        # 2 channels: [missing, hallucinated] logits
        self.out = nn.Conv3d(base, 2, kernel_size=1)

    def forward(self, batch):
        x = batch["input_tsdf"]  # [B,1,32,32,32]

        e1 = self.enc1(x)               # [B, b, 32,32,32]
        e2 = self.enc2(self.pool1(e1))  # [B,2b, 16,16,16]
        e3 = self.enc3(self.pool2(e2))  # [B,4b,  8, 8, 8]
        b  = self.bottleneck(self.pool3(e3))  # [B,8b,4,4,4]

        d3 = self.up3(b)                        # [B,4b,8,8,8]
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)                       # [B,2b,16,16,16]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)                       # [B,b,32,32,32]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.out(d1)                   # [B,2,32,32,32]
        miss_logits = logits[:, 0:1]
        hall_logits = logits[:, 1:2]

        miss = torch.sigmoid(miss_logits)
        hall = torch.sigmoid(hall_logits)

        # soft exclusivity so they don't fight
        hall = torch.clamp(hall, 0.0, 1.0)
        miss = torch.clamp(miss * (1.0 - hall), 0.0, 1.0)

        return {
            "mask_logits": logits,
            "mask_missing_logits": miss_logits,
            "mask_hallucinated_logits": hall_logits,
            "mask_missing": miss,
            "mask_hallucinated": hall,
        }



# ============================================================
# Small helpers
# ============================================================

class ConvGNAct3D(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        # safer than BN for small batch sizes
        num_groups = 8
        while out_c % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualConvBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct3D(channels, channels, 3, 1, 1),
            ConvGNAct3D(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.down = ConvGNAct3D(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.refine = ResidualConvBlock3D(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.refine(x)
        return x


class UpFuseBlock3D(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvGNAct3D(in_c + skip_c, out_c, 3, 1, 1),
            ResidualConvBlock3D(out_c),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return x


# ============================================================
# Transformer blocks
# ============================================================

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, drop: float = 0.0, bias: bool = True):
        super().__init__()
        hidden = hidden_features if hidden_features is not None else 4 * in_features
        self.fc1 = nn.Linear(in_features, hidden, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        fused_attn: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.attn_drop_p = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.fused_attn = fused_attn

        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=(self.attn_drop_p if self.training else 0.0),
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop_p, training=self.training)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
            bias=ffn_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# Ideal hybrid student
# ============================================================

class DinoDenseHybridStudent(nn.Module):
    """
    Hybrid conv + transformer student for dense voxel DINO features.

    Input:
      batch["input_tsdf"] : [B,1,32,32,32]

    Output:
      out["dino_proj"]    : [B,dino_dim,32,32,32]

    Optional:
      out["mask_logits"]  : [B,1,32,32,32]
      out["mask"]         : [B,1,32,32,32]
      out["x_norm_clstoken"] : [B,embed_dim]
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        ma = dict(config.get("model_args", {}))

        # top-level config style
        dino_dim = int(config.get("dino_dim", 1024))
        self.normalize_out = bool(config.get("normalize_out", False))

        # model_args-style config
        base_c = int(config.get("base_c", ma.get("base_c", 64)))              # conv backbone width
        embed_dim = int(ma.get("embed_dim", 256))                             # transformer dim
        decoder_hidden = int(ma.get("hidden_c", 256))                         # dense decoder width
        depth = int(ma.get("depth", 4))
        num_heads = int(ma.get("num_heads", 8))
        mlp_ratio = float(ma.get("mlp_ratio", 4.0))
        qkv_bias = bool(ma.get("qkv_bias", True))
        ffn_bias = bool(ma.get("ffn_bias", True))
        proj_bias = bool(ma.get("proj_bias", True))
        qk_norm = bool(ma.get("qk_norm", False))
        use_reentrant = bool(ma.get("use_reentrant", False))

        self.use_reentrant = use_reentrant
        self.embed_dim = embed_dim

        # -------------------------
        # Conv encoder
        # 32^3 -> 16^3 -> 8^3
        # -------------------------
        self.stem32 = nn.Sequential(
            ConvGNAct3D(1, base_c, 3, 1, 1),
            ResidualConvBlock3D(base_c),
        )  # [B, base_c, 32,32,32]

        self.down16 = DownBlock3D(base_c, base_c * 2)      # [B, 2b, 16,16,16]
        self.down8  = DownBlock3D(base_c * 2, base_c * 4)  # [B, 4b,  8, 8, 8]

        # project 8^3 conv features to transformer dim
        self.to_tokens = nn.Conv3d(base_c * 4, embed_dim, kernel_size=1)

        # learned CLS + pos
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 8 * 8 * 8, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # transformer on 8^3 = 512 tokens
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                ffn_bias=ffn_bias,
                proj_bias=proj_bias,
                drop=0.0,
                attn_drop=0.0,
                qk_norm=qk_norm,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                fused_attn=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # -------------------------
        # Dense decoder
        # transformer 8^3 -> 16^3 -> 32^3
        # -------------------------
        self.post_token = nn.Sequential(
            nn.Conv3d(embed_dim, decoder_hidden, kernel_size=1),
            nn.GELU(),
            ResidualConvBlock3D(decoder_hidden),
        )  # [B, hidden, 8,8,8]

        self.up16 = UpFuseBlock3D(
            in_c=decoder_hidden,
            skip_c=base_c * 2,
            out_c=decoder_hidden,
        )

        self.up32 = UpFuseBlock3D(
            in_c=decoder_hidden,
            skip_c=base_c,
            out_c=decoder_hidden,
        )

        self.refine32 = nn.Sequential(
            ResidualConvBlock3D(decoder_hidden),
            ResidualConvBlock3D(decoder_hidden),
        )

        self.dino_head = nn.Conv3d(decoder_hidden, dino_dim, kernel_size=1)

        # optional valid-voxel mask head
        self.use_mask_head = bool(ma.get("use_mask_head", True))
        if self.use_mask_head:
            self.mask_head = nn.Sequential(
                nn.Conv3d(decoder_hidden, decoder_hidden // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv3d(decoder_hidden // 2, 1, kernel_size=1),
            )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input_tsdf"]  # [B,1,32,32,32]
        B = x.shape[0]

        # -------------------------
        # Conv encoder
        # -------------------------
        s32 = self.stem32(x)     # [B,b,32,32,32]
        s16 = self.down16(s32)   # [B,2b,16,16,16]
        s8  = self.down8(s16)    # [B,4b, 8, 8, 8]

        # -------------------------
        # Tokens @ 8^3
        # -------------------------
        t = self.to_tokens(s8)                       # [B,embed,8,8,8]
        t = t.flatten(2).transpose(1, 2).contiguous()  # [B,512,embed]

        cls = self.cls_token.repeat(B, 1, 1)        # DDP-safe
        t = torch.cat([cls, t], dim=1)              # [B,513,embed]
        t = t + self.pos_embed[:, :t.shape[1]]

        for blk in self.blocks:
            if self.training:
                t = checkpoint(blk, t, use_reentrant=self.use_reentrant)
            else:
                t = blk(t)

        t = self.norm(t)
        cls_out = t[:, 0]                           # [B,embed]
        t = t[:, 1:]                                # [B,512,embed]

        # back to grid
        t = t.transpose(1, 2).contiguous().view(B, self.embed_dim, 8, 8, 8)

        # -------------------------
        # Dense decode with skips
        # -------------------------
        d8 = self.post_token(t)                     # [B,hidden,8,8,8]
        d16 = self.up16(d8, s16)                    # [B,hidden,16,16,16]
        d32 = self.up32(d16, s32)                   # [B,hidden,32,32,32]
        d32 = self.refine32(d32)                    # [B,hidden,32,32,32]

        pred = self.dino_head(d32)                  # [B,dino_dim,32,32,32]

        out = {
            "dino_proj": pred,
            "features": d32,
            "x_norm_clstoken": cls_out,
        }

        if self.use_mask_head:
            mask_logits = self.mask_head(d32)
            mask = torch.sigmoid(mask_logits)
            out["mask_logits"] = mask_logits
            out["mask"] = mask

        if self.normalize_out:
            out["dino_proj"] = F.normalize(out["dino_proj"], p=2, dim=1, eps=1e-6)

        return out
    


import math
from functools import partial
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================
# Reuse your existing helpers if already defined:
#   ConvGNAct3D
#   ResidualConvBlock3D
#   DownBlock3D
#   UpFuseBlock3D
#   Block
# ============================================================


class CrossAttentionTokens(nn.Module):
    """
    Cross-attention:
      query  : [B, Nq, C]
      memory : [B, Nm, C]
      output : [B, Nq, C]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        fused_attn: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.attn_drop_p = attn_drop

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        query  : [B, Nq, C]
        memory : [B, Nm, C]
        """
        B, Nq, C = query.shape
        Nm = memory.shape[1]

        q = self.q(query).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(memory).reshape(B, Nm, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(memory).reshape(B, Nm, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=(self.attn_drop_p if self.training else 0.0),
            )  # [B,H,Nq,D]
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # [B,H,Nq,Nm]
            attn = F.dropout(attn, p=self.attn_drop_p, training=self.training)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention block:
      x = x + CrossAttn(LN(x), LN(mem))
      x = x + MLP(LN(x))
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
    ):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_mem = norm_layer(dim)
        self.cross_attn = CrossAttentionTokens(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
            bias=ffn_bias,
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.norm_q(x), self.norm_mem(memory))
        x = x + self.mlp(self.norm2(x))
        return x

#old
# class DinoDenseHybridStudentPartTokens(nn.Module):
#     """
#     Part-token hybrid student.

#     Pipeline:
#       1) TSDF -> conv encoder -> 8^3 voxel features
#       2) project voxel features to tokens
#       3) prepend [CLS] and K learnable part tokens
#       4) joint self-attention on [CLS, PARTS, VOXELS]
#       5) decode voxel tokens using cross-attn from voxel queries to PART tokens
#       6) dense decoder back to 32^3
#       7) predict voxel DINO features

#     Outputs:
#       dino_proj           [B, dino_dim, 32,32,32]
#       features            [B, hidden,    32,32,32]
#       x_norm_clstoken     [B, embed_dim]
#       part_tokens         [B, K, embed_dim]
#       voxel_tokens        [B, 512, embed_dim]
#       part_logits_8       [B, K, 8,8,8]
#       part_probs_8        [B, K, 8,8,8]
#       mask_logits/mask    optional
#     """
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__()
#         self.config = config
#         ma = dict(config.get("model_args", {}))

#         dino_dim = int(config.get("dino_dim", 384))
#         self.normalize_out = bool(config.get("normalize_out", False))

#         base_c = int(config.get("base_c", ma.get("base_c", 64)))
#         embed_dim = int(ma.get("embed_dim", 256))
#         decoder_hidden = int(ma.get("hidden_c", 256))

#         depth = int(ma.get("depth", 4))
#         num_heads = int(ma.get("num_heads", 8))
#         mlp_ratio = float(ma.get("mlp_ratio", 4.0))
#         qkv_bias = bool(ma.get("qkv_bias", True))
#         ffn_bias = bool(ma.get("ffn_bias", True))
#         proj_bias = bool(ma.get("proj_bias", True))
#         qk_norm = bool(ma.get("qk_norm", False))
#         use_reentrant = bool(ma.get("use_reentrant", False))

#         self.num_parts = int(ma.get("num_parts", 8))
#         self.part_dropout = float(ma.get("part_dropout", 0.0))
#         self.use_part_decoder = bool(ma.get("use_part_decoder", True))
#         self.use_part_classifier = bool(ma.get("use_part_classifier", False))

#         self.use_mask_head = bool(ma.get("use_mask_head", True))
#         self.use_reentrant = use_reentrant
#         self.embed_dim = embed_dim

#         # ------------------------------------------------
#         # Conv encoder
#         # ------------------------------------------------
#         self.stem32 = nn.Sequential(
#             ConvGNAct3D(1, base_c, 3, 1, 1),
#             ResidualConvBlock3D(base_c),
#         )  # [B,b,32,32,32]

#         self.down16 = DownBlock3D(base_c, base_c * 2)      # [B,2b,16,16,16]
#         self.down8 = DownBlock3D(base_c * 2, base_c * 4)   # [B,4b,8,8,8]

#         # voxel -> token projection
#         self.to_tokens = nn.Conv3d(base_c * 4, embed_dim, kernel_size=1)

#         # ------------------------------------------------
#         # Learnable tokens
#         # ------------------------------------------------
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.part_tokens = nn.Parameter(torch.zeros(1, self.num_parts, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_parts + 8 * 8 * 8, embed_dim))

#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         nn.init.trunc_normal_(self.part_tokens, std=0.02)
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)

#         self.part_token_dropout = nn.Dropout(self.part_dropout)

#         # ------------------------------------------------
#         # Joint transformer over [CLS, PARTS, VOXELS]
#         # ------------------------------------------------
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 ffn_bias=ffn_bias,
#                 proj_bias=proj_bias,
#                 drop=0.0,
#                 attn_drop=0.0,
#                 qk_norm=qk_norm,
#                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                 fused_attn=True,
#             )
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

#         # ------------------------------------------------
#         # Part-aware voxel decoder
#         # voxel queries attend to refined part tokens
#         # ------------------------------------------------
#         if self.use_part_decoder:
#             self.part_decoder = CrossAttentionBlock(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 ffn_bias=ffn_bias,
#                 proj_bias=proj_bias,
#                 drop=0.0,
#                 attn_drop=0.0,
#                 qk_norm=qk_norm,
#                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                 fused_attn=True,
#             )
#         else:
#             self.part_decoder = None

#         # optional part assignment head at token level
        

#         # ------------------------------------------------
#         # Dense decoder
#         # ------------------------------------------------
#         self.post_token = nn.Sequential(
#             nn.Conv3d(embed_dim, decoder_hidden, kernel_size=1),
#             nn.GELU(),
#             ResidualConvBlock3D(decoder_hidden),
#         )

#         self.up16 = UpFuseBlock3D(
#             in_c=decoder_hidden,
#             skip_c=base_c * 2,
#             out_c=decoder_hidden,
#         )

#         self.up32 = UpFuseBlock3D(
#             in_c=decoder_hidden,
#             skip_c=base_c,
#             out_c=decoder_hidden,
#         )

#         self.refine32 = nn.Sequential(
#             ResidualConvBlock3D(decoder_hidden),
#             ResidualConvBlock3D(decoder_hidden),
#         )

#         self.dino_head = nn.Conv3d(decoder_hidden, dino_dim, kernel_size=1)

#         if self.use_mask_head:
#             self.mask_head = nn.Sequential(
#                 nn.Conv3d(decoder_hidden, decoder_hidden // 2, kernel_size=3, padding=1),
#                 nn.GELU(),
#                 nn.Conv3d(decoder_hidden // 2, 1, kernel_size=1),
#             )

#         if self.use_part_classifier:
#             self.part_classifier = nn.Linear(embed_dim, self.num_parts)
#         else:
#             self.part_classifier = None

#     def _pack_tokens(self, voxel_tokens: torch.Tensor) -> torch.Tensor:
#         """
#         voxel_tokens: [B, 512, C]
#         returns: [B, 1 + K + 512, C]
#         """
#         B = voxel_tokens.shape[0]
#         cls = self.cls_token.expand(B, -1, -1)
#         parts = self.part_tokens.expand(B, -1, -1)
#         parts = self.part_token_dropout(parts)
#         x = torch.cat([cls, parts, voxel_tokens], dim=1)
#         x = x + self.pos_embed[:, :x.shape[1]]
#         return x

#     def _unpack_tokens(self, x: torch.Tensor):
#         """
#         x: [B, 1+K+512, C]
#         """
#         cls = x[:, 0]                                      # [B,C]
#         parts = x[:, 1:1 + self.num_parts]                 # [B,K,C]
#         vox = x[:, 1 + self.num_parts:]                    # [B,512,C]
#         return cls, parts, vox

#     def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         x = batch["input_tsdf"]  # [B,1,32,32,32]
#         B = x.shape[0]

#         # ------------------------------------------------
#         # Conv encoder
#         # ------------------------------------------------
#         s32 = self.stem32(x)      # [B,b,32,32,32]
#         s16 = self.down16(s32)    # [B,2b,16,16,16]
#         s8 = self.down8(s16)      # [B,4b,8,8,8]

#         # ------------------------------------------------
#         # 8^3 voxel tokens
#         # ------------------------------------------------
#         vox = self.to_tokens(s8)                               # [B,C,8,8,8]
#         vox = vox.flatten(2).transpose(1, 2).contiguous()      # [B,512,C]

#         # ------------------------------------------------
#         # Joint self-attention with part tokens
#         # ------------------------------------------------
#         t = self._pack_tokens(vox)                             # [B,1+K+512,C]

#         for blk in self.blocks:
#             if self.training:
#                 t = checkpoint(blk, t, use_reentrant=self.use_reentrant)
#             else:
#                 t = blk(t)

#         t = self.norm(t)
#         cls_out, part_tokens, voxel_tokens = self._unpack_tokens(t)

#         # ------------------------------------------------
#         # Part-aware voxel decoding
#         # each voxel token attends to refined part tokens
#         # ------------------------------------------------
#         if self.part_decoder is not None:
#             voxel_tokens = self.part_decoder(voxel_tokens, part_tokens)

        

#         # ------------------------------------------------
#         # Back to dense grid
#         # ------------------------------------------------
#         voxel_grid = voxel_tokens.transpose(1, 2).contiguous().view(B, self.embed_dim, 8, 8, 8)

#         d8 = self.post_token(voxel_grid)                       # [B,h,8,8,8]
#         d16 = self.up16(d8, s16)                               # [B,h,16,16,16]
#         d32 = self.up32(d16, s32)                              # [B,h,32,32,32]
#         d32 = self.refine32(d32)                               # [B,h,32,32,32]

#         pred = self.dino_head(d32)                             # [B,dino_dim,32,32,32]

#         out = {
#             "dino_proj": pred,
#             "features": d32,
#             "x_norm_clstoken": cls_out,
#             "part_tokens": part_tokens,        # [B,K,C]
#             "voxel_tokens": voxel_tokens,      # [B,512,C]
#         }

#         # ------------------------------------------------
#         # Part assignment maps (debug / optional auxiliary loss)
#         # ------------------------------------------------
#         if self.use_part_classifier:
#             part_logits = self.part_classifier(voxel_tokens)       # [B,512,K]
#             part_logits = part_logits.transpose(1, 2).contiguous().view(B, self.num_parts, 8, 8, 8)
#             part_probs = F.softmax(part_logits, dim=1)
#             out["part_logits_8"] = part_logits
#             out["part_probs_8"] = part_probs
#         if self.use_mask_head:
#             mask_logits = self.mask_head(d32)
#             mask = torch.sigmoid(mask_logits)
#             out["mask_logits"] = mask_logits
#             out["mask"] = mask
#             out["dino_proj_wo_mask"] = pred
#             out["dino_proj"] = pred * mask

#         if self.normalize_out:
#             out["dino_proj"] = F.normalize(out["dino_proj"], p=2, dim=1, eps=1e-6)

#         return out

class DinoDenseHybridStudentPartTokens(nn.Module):
    """
    Part-token hybrid student.

    Pipeline:
      1) TSDF -> conv encoder -> 8^3 voxel features
      2) project voxel features to tokens
      3) prepend [CLS] and K learnable part tokens
      4) joint self-attention on [CLS, PARTS, VOXELS]
      5) decode voxel tokens using cross-attn from voxel queries to PART tokens
      6) dense decoder back to 32^3
      7) predict voxel DINO features

    Outputs:
      dino_proj           [B, dino_dim, 32,32,32]
      features            [B, hidden,    32,32,32]
      x_norm_clstoken     [B, embed_dim]
      part_tokens         [B, K, embed_dim]
      voxel_tokens        [B, 512, embed_dim]
      part_logits_8       [B, K, 8,8,8]
      part_probs_8        [B, K, 8,8,8]
      mask_logits/mask    optional
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        ma = dict(config.get("model_args", {}))

        dino_dim = int(config.get("dino_dim", 384))
        self.normalize_out = bool(config.get("normalize_out", False))

        base_c = int(config.get("base_c", ma.get("base_c", 64)))
        embed_dim = int(ma.get("embed_dim", 256))
        decoder_hidden = int(ma.get("hidden_c", 256))

        depth = int(ma.get("depth", 4))
        num_heads = int(ma.get("num_heads", 8))
        mlp_ratio = float(ma.get("mlp_ratio", 4.0))
        qkv_bias = bool(ma.get("qkv_bias", True))
        ffn_bias = bool(ma.get("ffn_bias", True))
        proj_bias = bool(ma.get("proj_bias", True))
        qk_norm = bool(ma.get("qk_norm", False))
        use_reentrant = bool(ma.get("use_reentrant", False))

        self.num_parts = int(ma.get("num_parts", 8))
        self.part_dropout = float(ma.get("part_dropout", 0.0))
        self.use_part_decoder = bool(ma.get("use_part_decoder", True))
        self.use_part_classifier = bool(ma.get("use_part_classifier", False))

        self.use_mask_head = bool(ma.get("use_mask_head", True))
        self.use_reentrant = use_reentrant
        self.embed_dim = embed_dim

        # ------------------------------------------------
        # Conv encoder
        # ------------------------------------------------
        self.stem32 = nn.Sequential(
            ConvGNAct3D(1, base_c, 3, 1, 1),
            ResidualConvBlock3D(base_c),
        )  # [B,b,32,32,32]

        self.down16 = DownBlock3D(base_c, base_c * 2)      # [B,2b,16,16,16]
        self.down8 = DownBlock3D(base_c * 2, base_c * 4)   # [B,4b,8,8,8]

        # voxel -> token projection
        self.to_tokens = nn.Conv3d(base_c * 4, embed_dim, kernel_size=1)

        # ------------------------------------------------
        # Learnable tokens
        # ------------------------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part_tokens = nn.Parameter(torch.zeros(1, self.num_parts, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_parts + 8 * 8 * 8, embed_dim))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.part_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.part_token_dropout = nn.Dropout(self.part_dropout)

        # ------------------------------------------------
        # Joint transformer over [CLS, PARTS, VOXELS]
        # ------------------------------------------------
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                ffn_bias=ffn_bias,
                proj_bias=proj_bias,
                drop=0.0,
                attn_drop=0.0,
                qk_norm=qk_norm,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                fused_attn=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # ------------------------------------------------
        # Part-aware voxel decoder
        # voxel queries attend to refined part tokens
        # ------------------------------------------------
        if self.use_part_decoder:
            self.part_decoder = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                ffn_bias=ffn_bias,
                proj_bias=proj_bias,
                drop=0.0,
                attn_drop=0.0,
                qk_norm=qk_norm,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                fused_attn=True,
            )
        else:
            self.part_decoder = None

        # optional part assignment head at token level
        

        # ------------------------------------------------
        # Dense decoder
        # ------------------------------------------------
        self.post_token = nn.Sequential(
            nn.Conv3d(embed_dim, decoder_hidden, kernel_size=1),
            nn.GELU(),
            ResidualConvBlock3D(decoder_hidden),
        )

        self.up16 = UpFuseBlock3D(
            in_c=decoder_hidden,
            skip_c=base_c * 2,
            out_c=decoder_hidden,
        )

        self.up32 = UpFuseBlock3D(
            in_c=decoder_hidden,
            skip_c=base_c,
            out_c=decoder_hidden,
        )

        self.refine32 = nn.Sequential(
            ResidualConvBlock3D(decoder_hidden),
            ResidualConvBlock3D(decoder_hidden),
        )

        self.dino_head = nn.Conv3d(decoder_hidden, dino_dim, kernel_size=1)

        if self.use_mask_head:
            self.mask_head = nn.Sequential(
                nn.Conv3d(decoder_hidden, decoder_hidden // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv3d(decoder_hidden // 2, 1, kernel_size=1),
            )

        if self.use_part_classifier:
            self.part_classifier = nn.Linear(embed_dim, self.num_parts)
        else:
            self.part_classifier = None

    def _pack_tokens(self, voxel_tokens: torch.Tensor) -> torch.Tensor:
        """
        voxel_tokens: [B, 512, C]
        returns: [B, 1 + K + 512, C]
        """
        B = voxel_tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        parts = self.part_tokens.expand(B, -1, -1)
        parts = self.part_token_dropout(parts)
        x = torch.cat([cls, parts, voxel_tokens], dim=1)
        x = x + self.pos_embed[:, :x.shape[1]]
        return x

    def _unpack_tokens(self, x: torch.Tensor):
        """
        x: [B, 1+K+512, C]
        """
        cls = x[:, 0]                                      # [B,C]
        parts = x[:, 1:1 + self.num_parts]                 # [B,K,C]
        vox = x[:, 1 + self.num_parts:]                    # [B,512,C]
        return cls, parts, vox

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["input_tsdf"]  # [B,1,32,32,32]
        B = x.shape[0]

        # ------------------------------------------------
        # Conv encoder
        # ------------------------------------------------
        s32 = self.stem32(x)      # [B,b,32,32,32]
        s16 = self.down16(s32)    # [B,2b,16,16,16]
        s8 = self.down8(s16)      # [B,4b,8,8,8]

        # ------------------------------------------------
        # 8^3 voxel tokens
        # ------------------------------------------------
        vox = self.to_tokens(s8)                               # [B,C,8,8,8]
        vox = vox.flatten(2).transpose(1, 2).contiguous()      # [B,512,C]

        # ------------------------------------------------
        # Joint self-attention with part tokens
        # ------------------------------------------------
        t = self._pack_tokens(vox)                             # [B,1+K+512,C]

        for blk in self.blocks:
            t = blk(t)

        t = self.norm(t)
        cls_out, part_tokens, voxel_tokens = self._unpack_tokens(t)

        # ------------------------------------------------
        # Part-aware voxel decoding
        # each voxel token attends to refined part tokens
        # ------------------------------------------------
        if self.part_decoder is not None:
            voxel_tokens = self.part_decoder(voxel_tokens, part_tokens)
                    # ------------------------------------------------
            # Direct voxel-to-part similarity (works even without part_classifier)
            # ------------------------------------------------
            voxel_tokens_norm = F.normalize(voxel_tokens, dim=-1)
            part_tokens_norm = F.normalize(part_tokens, dim=-1)

            part_logits_sim = voxel_tokens_norm @ part_tokens_norm.transpose(-1, -2)   # [B,512,K]
            part_probs_sim = F.softmax(part_logits_sim, dim=-1)                         # [B,512,K]

            part_logits_sim_8 = part_logits_sim.transpose(1, 2).contiguous().view(B, self.num_parts, 8, 8, 8)
            part_probs_sim_8 = part_probs_sim.transpose(1, 2).contiguous().view(B, self.num_parts, 8, 8, 8)

        

        # ------------------------------------------------
        # Back to dense grid
        # ------------------------------------------------
        voxel_grid = voxel_tokens.transpose(1, 2).contiguous().view(B, self.embed_dim, 8, 8, 8)

        d8 = self.post_token(voxel_grid)                       # [B,h,8,8,8]
        d16 = self.up16(d8, s16)                               # [B,h,16,16,16]
        d32 = self.up32(d16, s32)                              # [B,h,32,32,32]
        d32 = self.refine32(d32)                               # [B,h,32,32,32]

        pred = self.dino_head(d32)                             # [B,dino_dim,32,32,32]

        out = {
            "dino_proj": pred,
            "features": d32,
            "x_norm_clstoken": cls_out,
            "part_tokens": part_tokens,        # [B,K,C]
            "voxel_tokens": voxel_tokens,      # [B,512,C]
        }
        out["part_logits_sim_8"] = part_logits_sim_8
        out["part_probs_sim_8"] = part_probs_sim_8

        # ------------------------------------------------
        # Part assignment maps (debug / optional auxiliary loss)
        # ------------------------------------------------
        if self.use_part_classifier:
            part_logits = self.part_classifier(voxel_tokens)       # [B,512,K]
            part_logits = part_logits.transpose(1, 2).contiguous().view(B, self.num_parts, 8, 8, 8)
            part_probs = F.softmax(part_logits, dim=1)
            out["part_logits_8"] = part_logits
            out["part_probs_8"] = part_probs
        if self.use_mask_head:
            mask_logits = self.mask_head(d32)
            mask = torch.sigmoid(mask_logits)
            out["mask_logits"] = mask_logits
            out["mask"] = mask
            out["dino_proj_wo_mask"] = pred
            out["dino_proj"] = pred * mask

        if self.normalize_out:
            out["dino_proj"] = F.normalize(out["dino_proj"], p=2, dim=1, eps=1e-6)

        return out