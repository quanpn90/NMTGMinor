# Transformer networks for Neural Machine Translation

This is an implementation of the transformer for the paper

["Attention is all you need"](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

- Features supported:

+ Multi-layer transformer encoder-decoder networks
+ Multi-GPU / Single GPU training (mGPU is outdated for now)
+ Checkpointing models for better Memory/Speed trade-off
+ Research ideas 

- The code is based on several modules (Dictionary and Loss functions) of ["OpenNMT-py"](https://github.com/OpenNMT/OpenNMT-py)
