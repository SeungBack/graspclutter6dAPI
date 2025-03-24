# graspclutter6dAPI

- toolkit for GraspClutter6D Dataset
- load 6D grasp pose annotation, and grasp evaluation for benchmarking
- support compatibility with [graspnetAPI](https://github.com/graspnet/graspnetAPI) of GraspNet-1B

## Dataset

Dataset is available at [GraspClutter6D Website](https://huggingface.co/datasets/GraspClutter6D/graspclutter6d) via huggingface.

## Install

Install it via pip
```
pip install graspclutter6dAP
```

Or install it at local

```
conda create -n gc6d python=3.8
conda activate gc6d
pip install -e .

```

## Examples

```
# Set env
export GC6D_ROOT='/SSD1/graspclutter6d'

# Check the completeness of the data.
python3 exam_check_data.py

# How to load labels from GraspClutter6D
python3 exam_loadGrasp.py

# How to visualize the annotations
python3 exam_vis.py
```

## References

The codes of this repository are built upon the graspnetAPI. 
Thanks to the authors for sharing the code!