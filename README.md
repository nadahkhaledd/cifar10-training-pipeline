# CIFAR-10 Docker Training Pipeline

A **Dockerized PyTorch training pipeline** for CIFAR-10 image classification using PyTorch.
Supports **CPU and GPU execution**, multiple architectures, and reproducible environments.

The training script automatically selects:

- **GPU (CUDA)** if available
- **CPU** otherwise

---

## Features

- Fully Dockerized training environment
- CPU and GPU support
- Automatic device detection
- CIFAR-10 auto-download
- Multiple model architectures
- Optional quick subset training
- Model checkpoint saving
- Evaluation script included

---

## Supported Models

- `resnet18`
- `mobilenet_v2`
- `efficientnet_b0`

---

## Project Structure

model_training/

- Dockerfile
- Dockerfile.gpu
- requirements.txt
- train.py
- eval.py
- data/
- outputs/
- README.md

---

# CPU Setup

## Build Generic CPU Image
```bash
docker build -t cifar-trainer .
```
## Run Training
```bash
docker run --rm \
-v "$(pwd)/data:/app/data" \
-v "$(pwd)/outputs:/app/outputs" \
cifar-trainer \
python train.py --model mobilenet_v2 --epochs 1 --use_subset
```
---

# GPU Setup

## Build GPU Image

docker build -f Dockerfile.gpu -t cifar-trainer-gpu .

## Run GPU Training
```bash
docker run --rm --gpus all \
-v "$(pwd)/data:/app/data" \
-v "$(pwd)/outputs:/app/outputs" \
cifar-trainer-gpu \
python train.py --model mobilenet_v2 --epochs 1 --use_subset
```
---

# Building Separate Images Per Model (Optional)

If you want **three dedicated images**, one per model, you can tag them differently.

## Build Images

ResNet18:
```bash
docker build -t cifar-resnet18 .
```
MobileNetV2:
```bash
docker build -t cifar-mobilenet_v2 .
```
EfficientNet-B0:
```bash
docker build -t cifar-efficientnet_b0 .
```
---

# Running Each Model (CPU)

## ResNet18
```bash
docker run --rm \
-v "$(pwd)/data:/app/data" \
-v "$(pwd)/outputs:/app/outputs" \
cifar-resnet18 \
python train.py --model resnet18 --epochs 3 --use_subset
```
## MobileNetV2
```bash
docker run --rm \
-v "$(pwd)/data:/app/data" \
-v "$(pwd)/outputs:/app/outputs" \
cifar-mobilenet_v2 \
python train.py --model mobilenet_v2 --epochs 3 --use_subset
```
## EfficientNet-B0
```bash
docker run --rm \
-v "$(pwd)/data:/app/data" \
-v "$(pwd)/outputs:/app/outputs" \
cifar-efficientnet_b0 \
python train.py --model efficientnet_b0 --epochs 3 --use_subset
```
---

# Running Each Model (GPU)

## ResNet18
```bash
docker run --rm --gpus all \
-v "$(pwd)/data:/app/data" \
-v "$(pwd)/outputs:/app/outputs" \
cifar-trainer-gpu \
python train.py --model resnet18 --epochs 3 --use_subset
```
## MobileNetV2
```bash
docker run --rm --gpus all \
-v "$(pwd)/data:/app/data" \
-v "$(pwd)/outputs:/app/outputs" \
cifar-trainer-gpu \
python train.py --model mobilenet_v2 --epochs 3 --use_subset
```
## EfficientNet-B0
```bash
docker run --rm --gpus all \
-v "$(pwd)/data:/app/data" \
-v "$(pwd)/outputs:/app/outputs" \
cifar-trainer-gpu \
python train.py --model efficientnet_b0 --epochs 3 --use_subset
```
---

# Training Modes

## Quick Test
```bash
python train.py \
--model mobilenet_v2 \
--epochs 1 \
--use_subset
```
Subset sizes:

- Train: 5000
- Test: 1000

## Full Training
```bash
python train.py \
--model mobilenet_v2 \
--epochs 10
```
---

# Outputs

Saved models:

outputs/<model_name>_best.pth

Examples:

outputs/resnet18_best.pth  
outputs/mobilenet_v2_best.pth  
outputs/efficientnet_b0_best.pth  

---

# Evaluation
```bash
docker run --rm \
-v "$(pwd)/data:/app/data" \
-v "$(pwd)/outputs:/app/outputs" \
cifar-trainer \
python eval.py --model mobilenet_v2 --weights outputs/mobilenet_v2_best.pth
```
---

# Dataset Handling

Dataset is:

- Downloaded automatically
- Cached locally
- Reused across runs

If already present:

Dataset found — skipping download

---

# Reproducibility

Docker ensures:

- Same Python version
- Same dependency versions
- Same training behavior

Across machines.

---

# Requirements

## CPU

Docker Desktop

## GPU

- NVIDIA GPU
- NVIDIA drivers
- NVIDIA Container Toolkit

