# Feature Gradient Gating for Vision Transformers

Research investigating whether SAE (Sparse Autoencoder) feature gradients can improve attribution maps by extending the TransMM principle from attention space into interpretable feature space.

**Project Page**: [https://piragi.github.io/thesis](https://piragi.github.io/thesis)

## Overview

Traditional attribution methods like TransMM combine attention maps with gradients to generate explanations. This project explores whether this principle extends to SAE feature space:

- **Conventional TransMM**: Combines attention patterns with attention gradients in attention space
- **Feature Gradient Gating**: Combines SAE feature activations with SAE feature gradients in interpretable feature space

The key research question: Can we leverage the interpretability of SAE features to create more faithful and semantically meaningful attribution maps?

## Setup

### Prerequisites

Using `uv` is recommended: https://docs.astral.sh/uv/

A Hugging Face token is required to download the ImageNet dataset. Login with:

```bash
uvx hf auth login
```

### Installation

Run the setup script:

```bash
uv run setup.py
```

This will download all datasets, models, and SAE checkpoints.

## Usage

### Running Experiments

Run the main experiment sweep:

```bash
uv run main.py
```

### Datasets

The project supports three medical/natural image datasets:
- **ImageNet**: Natural images (1000 classes)
- **HyperKvasir**: Gastrointestinal tract images
- **CovidQUEx**: Lung X-ray images
