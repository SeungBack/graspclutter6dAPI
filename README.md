# GraspClutter6D API

<div align="center">
  <img src="./example.jpg" alt="GraspClutter6D example visualization">
</div>

## Overview

This repository contains dataset toolkit for the paper "**GraspClutter6D: A Large-scale Real-world Dataset for Robust Perception and Grasping in Cluttered Scenes**." (RA-L 2025) 

[[ArXiv]](https://arxiv.org/abs/2504.06866) [[Website]](https://sites.google.com/view/graspclutter6d) [[Dataset]](https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D) [[Video]](https://youtu.be/NkKkfVS5wZ4)

- Load and manipulate 6D grasp pose annotations
- Perform grasp evaluation for benchmarking
- Maintain compatibility with [graspnetAPI](https://github.com/graspnet/graspnetAPI) from GraspNet-1B

## Dataset

The GraspClutter6D Dataset is available through [Hugging Face](https://huggingface.co/datasets/GraspClutter6D/graspclutter6d)

## Installation

### Option 1: Install from source (Recommended for latest version)

```bash
# Create and activate conda environment
conda create -n gc6d python=3.8
conda activate gc6d

# Clone repository and install in development mode
pip install git+https://github.com/seungback/graspnetAPI.git
```

### Option 2: Install via pip (Deprecated)

```bash
pip install graspclutter6dAPI
```

## Environment Setup

Before running the examples, set the environment variable to point to your dataset location:

```bash
export GC6D_ROOT='/path/to/graspclutter6d'
```

## Usage Examples

### Validate Dataset Integrity

Check if the downloaded dataset is complete and properly structured:

```bash
python examples/exam_check_data.py
```

### Load Grasp Annotations

Load and process grasp annotations from the GraspClutter6D dataset:

```bash
python examples/exam_loadGrasp.py
```

### Visualize Grasp Annotations

Generate visualizations of grasp poses and objects:

```bash
python examples/exam_vis.py
```

## Acknowledgments

This repository is built upon the [graspnetAPI](https://github.com/graspnet/graspnetAPI). We thank the authors for sharing their code.

