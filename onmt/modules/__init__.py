from onmt.modules.attention import MultiHeadAttention
from onmt.modules.base_seq2seq import Generator, NMTModel
from onmt.modules.static_dropout import StaticDropout

# For flake8 compatibility.
__all__ = [MultiHeadAttention, Generator, NMTModel, StaticDropout]
