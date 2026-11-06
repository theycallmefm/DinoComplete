import torch
import torch.nn as nn
import math
from models.blocks import BasicBlock, MultiScaleChunkVoxelMamba3D, DecoderDualVoxelMamba3D

class DinoComplete(nn.Module):
    """
    BaselineMultiScaleChunkVoxelMambaDualDecoder + DINO conditioning.

    Difference from the original model:
      - The original uses:
            Fe16 = self._decoder_mamba_16(f16, d16)
      - This version uses:
            Fe16 = self._decoder_mamba_16(dino16, d16)

    Pipeline:
      1) TSDF -> encoder_32 -> msm_32 -> encoder_16 -> encoder_8 -> msm_8
      2) Decoder: 8^3 -> 16^3 gives d16
      3) DINO model extracts voxel features from input TSDF
      4) DINO features are adapted to [B, C, 16,16,16]
      5) decoder_mamba_16(dino16, d16)
      6) 16^3 -> 32^3 -> final TSDF
    """

    def __init__(self, args, device, dino_model):
        super().__init__()

        self._device = device
        self._channel_num = int(args.channel_num)
        self._truncation = float(args.truncation)

        C = self._channel_num

        # -------------------------------------------------
        # DINO MODEL SETUP
        # -------------------------------------------------
        self.dino_model = dino_model.to(device)

        self.freeze_dino = bool(getattr(args, "freeze_dino", True))
        if self.freeze_dino:
            self.dino_model.eval()
            for p in self.dino_model.parameters():
                p.requires_grad_(False)

        self.dino_input_key = getattr(args, "dino_input_key", "input_tsdf")
        self.dino_feat_key = getattr(args, "dino_feat_key", "features")

        # channel dim of dino_model output tensor
        # for example 64 / 128 / 512 / 768 / 1024 depending on your setup
        self.dino_out_dim = int(
            getattr(
                args,
                "dino_out_dim",
                getattr(args, "dino_channel_dim", C)
            )
        )

        # expected DINO voxel resolution before adaptation
        self.dino_input_resolution = int(getattr(args, "dino_input_resolution", 32))

        # target resolution for decoder_mamba_16
        self.dino_target_resolution = 16

        # adapt DINO features to exactly [B, C, 16,16,16]
        self.dino_adapter_16 = self._make_dino_adapter(
            c_in=self.dino_out_dim,
            c_out=C,
            input_resolution=self.dino_input_resolution,
            target_resolution=self.dino_target_resolution,
        )

        # -------------------------------------------------
        # ORIGINAL MODEL
        # -------------------------------------------------

        # -----------------------
        # ENCODER STAGE 0: 32^3
        # -----------------------
        self._encoder_32 = BasicBlock(
            'Encoder',
            in_channels=1,
            out_channels=C,
            kernel_size=3,
            stride=1,
        )  # [B, C, 32,32,32]



        # -----------------------
        # ENCODER STAGE 1: 16^3
        # -----------------------
        self._encoder_16 = BasicBlock(
            'Encoder',
            in_channels=C,
            out_channels=C,
            kernel_size=3,
            stride=2,
        )  # [B, C, 16,16,16]

        # -----------------------
        # ENCODER STAGE 2: 8^3
        # -----------------------
        self._encoder_8 = BasicBlock(
            'Encoder',
            in_channels=C,
            out_channels=C,
            kernel_size=3,
            stride=2,
        )  # [B, C, 8,8,8]


        # -----------------------
        # DECODER STAGE 1: 8^3 -> 16^3
        # -----------------------
        self._decoder_1 = BasicBlock(
            'Decoder',
            in_channels=C,
            out_channels=C,
            kernel_size=3,
            stride=2,
        )  # [B, C, 16,16,16]

        # -----------------------
        # DECODER MAMBA at 16^3
        # -----------------------
        self._decoder_mamba_16 = DecoderDualVoxelMamba3D(
            channels=C,
            grid_size=(16, 16, 16),
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # -----------------------
        # DECODER STAGE 2: 16^3 -> 32^3
        # -----------------------
        self._decoder_2 = BasicBlock(
            'Decoder',
            in_channels=C,
            out_channels=C,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )  # [B, C, 32,32,32]

        self._msm_32 = MultiScaleChunkVoxelMamba3D(
            channels=C,
            grid_size=(32, 32, 32),
            d_state=16,
            d_conv=4,
            expand=2,
            window_size=(4, 4, 4),
            chunk_sizes=(8, 4),
        )

        # -----------------------
        # FINAL TSDF PREDICTOR
        # -----------------------
        self._conv_last = nn.Conv3d(C, 1, kernel_size=3, stride=1, padding=1)

    def _make_dino_adapter(self, c_in: int, c_out: int, input_resolution: int = 32, target_resolution: int = 16):
        """
        Adapts DINO voxel features to the channel count and spatial resolution
        expected by decoder_mamba_16.

        Output shape:
            [B, c_out, target_resolution, target_resolution, target_resolution]
        """
        assert input_resolution >= target_resolution, "input_resolution must be >= target_resolution"

        # choose GroupNorm groups safely
        gn_groups = min(32, c_out)
        while c_out % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1

        layers = [
            nn.Conv3d(c_in, c_out, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, c_out),
            nn.SiLU(inplace=True),
        ]

        if input_resolution == target_resolution:
            pass
        elif input_resolution > target_resolution:
            scale = input_resolution // target_resolution
            assert input_resolution % target_resolution == 0, "target_resolution must divide input_resolution"
            assert (scale & (scale - 1)) == 0, "downsampling ratio must be a power of 2"

            num_down = int(math.log2(scale))
            for _ in range(num_down):
                layers += [
                    nn.Conv3d(c_out, c_out, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(gn_groups, c_out),
                    nn.SiLU(inplace=True),
                ]
        else:
            scale = target_resolution // input_resolution
            assert target_resolution % input_resolution == 0, "input_resolution must divide target_resolution"
            assert (scale & (scale - 1)) == 0, "upsampling ratio must be a power of 2"

            num_up = int(math.log2(scale))
            for _ in range(num_up):
                layers += [
                    nn.ConvTranspose3d(c_out, c_out, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.GroupNorm(gn_groups, c_out),
                    nn.SiLU(inplace=True),
                ]

        return nn.Sequential(*layers)

    def _extract_dino_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw DINO voxel features.
        Expected shape before adapter:
            [B, dino_out_dim, R, R, R]
        """
        if self.freeze_dino:
            with torch.no_grad():
                dino_feat = self.dino_model({self.dino_input_key: x})[self.dino_feat_key]
        else:
            dino_feat = self.dino_model({self.dino_input_key: x})[self.dino_feat_key]

        return dino_feat.to(self._device)

    def forward(self, x):
        # x: [B,1,32,32,32]
        # -----------------------
        # ENCODER @ 32^3 + chunkwise (8,4)
        # -----------------------
        f32 = self._encoder_32(x)        # [B, C,32,32,32]
        #f32 = self._msm_32(f32)          # [B, C,32,32,32]

        # -----------------------
        # ENCODER down to 16^3, 8^3
        # -----------------------
        f16 = self._encoder_16(f32)      # [B, C,16,16,16]
        f8  = self._encoder_8(f16)       # [B, C, 8, 8, 8]

        # -----------------------
        # BACKBONE: multi-scale chunkwise at 8^3 (4,2)
        # -----------------------

        # -----------------------
        # DECODER: 8 -> 16
        # -----------------------
        d16 = self._decoder_1(f8)        # [B, C,16,16,16]

        # -----------------------
        # DINO FEATURES -> adapt to [B,C,16,16,16]
        # -----------------------
        dino_feat = self._extract_dino_features(x)
        #

        #dino_feat = input_dict["in_dino"].to(self._device, non_blocking=True)  # [B,1,32,32,32]
        dino16 = self.dino_adapter_16(dino_feat)

        # -----------------------
        # Voxel State Operator: TSDF + DINO + d16 at 16^3
        # replaces original:
        #   Fe16 = self._decoder_mamba_16(f16, d16)
        # -----------------------
        Fe16 = self._decoder_mamba_16(dino16, d16)  # [B, C,16,16,16]

        # -----------------------
        # DECODER: 16 -> 32
        # -----------------------
        d32 = self._decoder_2(Fe16)      # [B, C,32,32,32]

        d32 = self._msm_32(d32)          # [B, C,32,32,32]

        # Final TSDF
        out = self._conv_last(d32)       # [B,1,32,32,32]

        return {
            "output": out,
        }

