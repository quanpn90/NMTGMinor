from onmt.modules.GlobalAttention import GlobalAttention
from onmt.modules.BaseModel import Generator, NMTModel
from onmt.modules.StaticDropout import StaticDropout

# For flake8 compatibility.
__all__ = [GlobalAttention, Generator, NMTModel, StaticDropout]
