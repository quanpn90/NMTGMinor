# Introduction

# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.7+

Currently NMTG requires PyTorch version >= 1.8.0. Best is 1.10.0
Please follow the instructions here: https://github.com/pytorch/pytorch#installation.


After PyTorch is installed, you can install the requirements with:
```
pip install -r requirements.txt
```

# C++/CUDA module installation

NMTG supports a couple of modules written using custom Pytorch/C++/CUDA modules to utilize GPU better and reduce overheads, including:
* Self-attention and encoder-decoder attention with CUBLASLT
* Multi-layer Perceptrons with CUBLASLT and fused dropout-relu/gelu/silu where inplace is implemented whenever possible
* Highly optimized layer norm and multi-head attention (only available with sm80 (NVIDIA A100)) from Apex
* Fused Logsoftmax/Cross-entropy loss to save memory for large output layer, from Apex
* Fused inplaced Dropout Add for residual Transformers

Installation requires CUDA and nvcc with the same version with PyTorch. Its possible to install CUDA from conda via:

```
conda install -c nvidia/label/cuda-11.5.2 cuda-toolkit
```

And then navigate to the extension modules and install nmtgminor-cuda via

```
cd onmt/modules/extension
python setup.py install
```

Without this step, all modules backoff to PyTorch versions.

# IWSLT 2022 Speech Translation models


# Interspeech 2022 Multilingual ASR models 