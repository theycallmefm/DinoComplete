# DinoComplete: 3D Shape Completion with Distilled Semantic Priors and State Space Models

TSDF-based 3D shape completion framework with distilleTSDF-DINO feature priors.

# Installation

Create the environment using:

```bash
conda env create -f environment.yaml
conda activate dino-complete
```

After environment installation, install the `mamba-ssm` library separately. We followed the installation instructions provided [here](https://github.com/gwenzhang/Voxel-Mamba/blob/master/docs/INSTALL.md)




# Dataset

This project uses the same dataset structure as: [PatchComplete](https://github.com/yuchenrao/PatchComplete):

Please prepare the ShapeNet and ScanNet datasets by following the PatchComplete instructions.

The default dataset path is:

```text
../PatchComplete/data/
```

You can modify the dataset path in the configuration files if needed.

# TSDF-DINO Distillation

Instructions for TSDF-DINO distillation are provided in:

```text
models/tsdf-dino/
```

We also provide a pretrained TSDF-DINO model under:

```text
models/tsdf-dino/pretrained_model/
```

To use TSDF-DINO priors during shape completion training, set `dino_priors` to the corresponding checkpoint path, as shown in the default YAML configuration files.

# Training

Example YAML configuration files are provided under:

```text
configs/
```

Modify the parameters according to your setup and requirements.

## ShapeNet Training

```bash
python main.py --config configs/shapenet_train.yaml
```

## ScanNet Fine-tuning

```bash
python main.py --config configs/scannet_finetune.yaml
```

# Evaluation and Generation

We provide pretrained shape completion models under:

```text
pretrained_models/
```

## Evaluation

Run evaluation using:

```bash
python generation.py --config configs/evaluation.yaml
```

## Sample Generation

Generate samples using our pretrained model:

```bash
python generation.py --config configs/sample_generation.yaml
```

# Project Structure

```text
DinoComplete/
├── configs/
│   ├── shapenet_train.yaml
│   ├── scannet_finetune.yaml
│   ├── evaluation.yaml
│   └── sample_generation.yaml
│
├── models/
│   └── tsdf-dino/
│       ├── pretrained_model/
│       └── README.md
│
├── pretrained_models/
│
├── main.py
├── generation.py
├── environment.yaml
└── README.md
```

# Environment

Our experiments were conducted using the following setup:

```text
CUDA Toolkit: 11.8
NVIDIA Driver: 525.85.05
GPU: NVIDIA A40
```

