from functools import partial
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct3D(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_c,
            out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

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
        x = F.interpolate(
            x,
            size=skip.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return x


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
        self.fused_attn = fused_attn
        self.attn_drop_p = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop_p if self.training else 0.0,
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
        B, Nq, C = query.shape
        Nm = memory.shape[1]

        q = self.q(query)
        q = q.reshape(B, Nq, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        k = self.k(memory)
        k = k.reshape(B, Nm, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)

        v = self.v(memory)
        v = v.reshape(B, Nm, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop_p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
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