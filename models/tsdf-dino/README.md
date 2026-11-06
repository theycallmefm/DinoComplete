# TSDF-DINO Distillation

This repository contains the code for our TSDF-DINO distillation framework.

# Dataset

First, download the dataset from [PatchComplete](https://github.com/yuchenrao/PatchComplete):

Prepare the ShapeNet and ScanNet datasets by following the PatchComplete instructions.

The default dataset path is:

```text
../PatchComplete/data/
```


# Rendering ShapeNet Objects

For the preprocessing step, rendered RGB images and depth maps are required for each ShapeNet object.

First, install the depth renderer:

```bash
git clone https://github.com/yinyunie/depth_renderer.git
```

Generate 20 rendered views and depth maps for each ShapeNet object using the default settings provided by the repository.

Your rendering directory should have the following structure:

```text
depth_renderer/
└── datasets/
    ├── ShapeNetRenderings/
    │   └── 02801938/
    │       └── ab6e1ccf0f35228416fdce62f5c06e29/
    │           ├── color_001.png
    │           ├── color_002.png
    │           ├── ...
    │           ├── color_020.png
    │           ├── depth_001.exr
    │           ├── depth_002.exr
    │           ├── ...
    │           └── depth_020.exr
    │
    └── camera_settings/
        ├── cam_K/
        │   └── cam_K.txt
        │
        └── cam_RT/
            ├── cam_RT_001.txt
            ├── cam_RT_002.txt
            ├── ...
            └── cam_RT_020.txt
```

# Installing DINOv3

Install DINOv3 by following the instructions from:

- https://github.com/facebookresearch/dinov3.git

We used the following pretrained model for our experiments:

```text
dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

You may use other pretrained models if desired, but you must adjust the channel dimensions in the configuration files accordingly.

# Precomputing Ground-Truth DINO Volumes

After installing everything, create fused DINO volumes using:

```bash
python data/precompute_gt_dino_features.py
```

Example command:

```bash
python precompute_gt_dino_features.py \
--render_root ../../depth_renderer/datasets/ShapeNetRenderings \
--tsdf_root ../../PatchComplete/data/shapenet \
--camera_root ../../depth_renderer/datasets/camera_settings \
--list_file ../../data/data_splits/shapenet_test.txt \
--model dinov3_vits16 \
--grid_sizes 32
```

This will save fused DINO feature volumes for each sample in the provided list.

# Generating Incomplete DINO Features

Afterward, generate incomplete DINO features using:

```bash
python precompute_incompletedino_from_gtdino.py \
--render_root ../../../depth_renderer/datasets/ShapeNetRenderings \
--tsdf_root ../../../PatchComplete/data/shapenet \
--camera_root ../../../depth_renderer/datasets/camera_settings \
--list_file ../../data/data_splits/shapenet_test.txt \
--grid_sizes 32 \
--views 5 6 7 8
```

# ShapeNet TSDF Folder Structure

After preprocessing, each ShapeNet object folder should have the following structure:

```text
PatchComplete/
└── data/
    └── shapenet/
        └── 02801938/
            └── ab6e1ccf0f35228416fdce62f5c06e29/
                ├── dinov3_vits16_4_32_inside.npz
                ├── dinov3_vits16_5_32_inside.npz
                ├── dinov3_vits16_6_32_inside.npz
                ├── dinov3_vits16_7_32_inside.npz
                ├── dinov3_vits16_gt_32_inside.npz
                ├── gt.npz
                ├── gt.off
                ├── input_4.npz
                ├── input_5.npz
                ├── input_6.npz
                └── input_7.npz
```

# Training

You can find an example YAML configuration file in:

```text
config_distillation.yaml
```

Modify the parameters according to your setup and requirements.

Run training with:

```bash
python main.py --config config_distillation.yaml
```


# Pretrained Models

We also provide a pretrained TSDF-DINO model under:

```text
pretrained_model/
```

To use TSDF-DINO priors during shape completion training, set `dino_priors` to the corresponding checkpoint path, as shown in the default YAML configuration files.

# Environment

Our experiments were conducted using the following setup:

```text
CUDA Toolkit: 11.8
NVIDIA Driver: 525.85.05
GPU: NVIDIA A40
```

