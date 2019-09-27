from onmt.modules.GlobalAttention import MultiHeadAttention
from onmt.modules.BaseModel import Generator, NMTModel
from onmt.modules.StaticDropout import StaticDropout

# For flake8 compatibility.
__all__ = [MultiHeadAttention, Generator, NMTModel, StaticDropout]
