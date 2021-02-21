"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import fnmatch
import json
import logging
import os
import re
import shutil
import sys
import tarfile
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields
from functools import partial, wraps
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile

import numpy as np
import requests
# from filelock import FileLock
from tqdm.auto import tqdm

# from . import __version__


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
    if USE_TORCH in ("1", "ON", "YES", "AUTO") and USE_TF not in ("1", "ON", "YES"):
        import torch

        _torch_available = True  # pylint: disable=invalid-name
        logger.info("PyTorch version {} available.".format(torch.__version__))
    else:
        logger.info("Disabling PyTorch because USE_TF is set")
        _torch_available = False
except ImportError:
    _torch_available = False  # pylint: disable=invalid-name

try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

    if USE_TF in ("1", "ON", "YES", "AUTO") and USE_TORCH not in ("1", "ON", "YES"):
        import tensorflow as tf

        assert hasattr(tf, "__version__") and int(tf.__version__[0]) >= 2
        _tf_available = True  # pylint: disable=invalid-name
        logger.info("TensorFlow version {} available.".format(tf.__version__))
    else:
        logger.info("Disabling Tensorflow because USE_TORCH is set")
        _tf_available = False
except (ImportError, AssertionError):
    _tf_available = False  # pylint: disable=invalid-name


try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )


try:
    import torch_xla.core.xla_model as xm  # noqa: F401

    if _torch_available:
        _torch_tpu_available = True  # pylint: disable=
    else:
        _torch_tpu_available = False
except ImportError:
    _torch_tpu_available = False


try:
    import psutil  # noqa: F401

    _psutil_available = True

except ImportError:
    _psutil_available = False


try:
    import py3nvml  # noqa: F401

    _py3nvml_available = True

except ImportError:
    _py3nvml_available = False



try:
    from apex import amp  # noqa: F401

    _has_apex = True
except ImportError:
    _has_apex = False

default_cache_path = os.path.join(torch_cache_home, "transformers")


PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
CONFIG_NAME = "config.json"
MODEL_CARD_NAME = "modelcard.json"


MULTIPLE_CHOICE_DUMMY_INPUTS = [[[0], [1]], [[0], [1]]]
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]

S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available


def is_torch_tpu_available():
    return _torch_tpu_available


def is_psutil_available():
    return _psutil_available


def is_py3nvml_available():
    return _py3nvml_available


def is_apex_available():
    return _has_apex


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_callable(*docstr):
    def docstring_decorator(fn):
        class_name = ":class:`~transformers.{}`".format(fn.__qualname__.split(".")[0])
        intro = "   The {} forward method, overrides the :func:`__call__` special method.".format(class_name)
        note = r"""
    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        pre and post processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + "".join(docstr)
        return fn

    return docstring_decorator


RETURN_INTRODUCTION = r"""
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(torch.FloatTensor)`:
        A :class:`~{full_output_type}` (if ``return_dict=True`` is passed or when ``config.return_dict=True``) or a
        tuple of :obj:`torch.FloatTensor` comprising various elements depending on the configuration
        (:class:`~transformers.{config_class}`) and inputs.
"""


def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # Split output_arg_doc in blocks argument/description
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    for line in output_args_doc.split("\n"):
        # If the indent is the same as the beginning, the line is the name of new arg.
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            # Otherwise it's part of the description of the current arg.
            # We need to remove 2 spaces to the indentation.
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    # Format each block for proper rendering
    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    return "\n".join(blocks)


def _prepare_output_docstrings(output_type, config_class):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    docstrings = output_type.__doc__

    # Remove the head of the docstring to keep the list of args only
    lines = docstrings.split("\n")
    i = 0
    while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
        i += 1
    if i < len(lines):
        docstrings = "\n".join(lines[(i + 1) :])
        docstrings = _convert_output_args_doc(docstrings)

    # Add the return introduction
    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    intro = RETURN_INTRODUCTION.format(full_output_type=full_output_type, config_class=config_class)
    return intro + docstrings


PT_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_QUESTION_ANSWERING_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])
        >>> outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_scores
        >>> end_scores = outputs.end_scores
"""

PT_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_MASKED_LM_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)
        >>> input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
        >>> outputs = model(input_ids, labels=input_ids)
        >>> loss = outputs.loss
        >>> prediction_logits = outputs.logits
"""

PT_BASE_MODEL_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
"""

PT_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)
        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."
        >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
        >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
        >>> outputs = model(**{{k: v.unsqueeze(0) for k,v in encoding.items()}}, labels=labels)  # batch size is 1
        >>> # the linear classifier still needs to be trained
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_CAUSAL_LM_SAMPLE = r"""
    Example::
        >>> import torch
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs, labels=inputs["input_ids"])
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> input_ids = inputs["input_ids"]
        >>> inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1
        >>> outputs = model(inputs)
        >>> loss, scores = outputs[:2]
"""

TF_QUESTION_ANSWERING_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')
        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> input_dict = tokenizer(question, text, return_tensors='tf')
        >>> start_scores, end_scores = model(input_dict)
        >>> all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        >>> answer = ' '.join(all_tokens[tf.math.argmax(start_scores, 1)[0] : tf.math.argmax(end_scores, 1)[0]+1])
"""

TF_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1
        >>> outputs = model(inputs)
        >>> loss, logits = outputs[:2]
"""

TF_MASKED_LM_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')
        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
        >>> outputs = model(input_ids)
        >>> prediction_scores = outputs[0]
"""

TF_BASE_MODEL_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)
        >>> last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
"""

TF_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')
        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."
        >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='tf', padding=True)
        >>> inputs = {{k: tf.expand_dims(v, 0) for k, v in encoding.items()}}
        >>> outputs = model(inputs)  # batch size is 1
        >>> # the linear classifier still needs to be trained
        >>> logits = outputs[0]
"""

TF_CAUSAL_LM_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf
        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)
        >>> logits = outputs[0]
"""


def add_code_sample_docstrings(*docstr, tokenizer_class=None, checkpoint=None, output_type=None, config_class=None):
    def docstring_decorator(fn):
        model_class = fn.__qualname__.split(".")[0]
        is_tf_class = model_class[:2] == "TF"

        if "SequenceClassification" in model_class:
            code_sample = TF_SEQUENCE_CLASSIFICATION_SAMPLE if is_tf_class else PT_SEQUENCE_CLASSIFICATION_SAMPLE
        elif "QuestionAnswering" in model_class:
            code_sample = TF_QUESTION_ANSWERING_SAMPLE if is_tf_class else PT_QUESTION_ANSWERING_SAMPLE
        elif "TokenClassification" in model_class:
            code_sample = TF_TOKEN_CLASSIFICATION_SAMPLE if is_tf_class else PT_TOKEN_CLASSIFICATION_SAMPLE
        elif "MultipleChoice" in model_class:
            code_sample = TF_MULTIPLE_CHOICE_SAMPLE if is_tf_class else PT_MULTIPLE_CHOICE_SAMPLE
        elif "MaskedLM" in model_class:
            code_sample = TF_MASKED_LM_SAMPLE if is_tf_class else PT_MASKED_LM_SAMPLE
        elif "LMHead" in model_class:
            code_sample = TF_CAUSAL_LM_SAMPLE if is_tf_class else PT_CAUSAL_LM_SAMPLE
        elif "Model" in model_class:
            code_sample = TF_BASE_MODEL_SAMPLE if is_tf_class else PT_BASE_MODEL_SAMPLE
        else:
            raise ValueError(f"Docstring can't be built for model {model_class}")

        output_doc = _prepare_output_docstrings(output_type, config_class) if output_type is not None else ""
        built_doc = code_sample.format(model_class=model_class, tokenizer_class=tokenizer_class, checkpoint=checkpoint)
        fn.__doc__ = (fn.__doc__ or "") + "".join(docstr) + output_doc + built_doc
        return fn

    return docstring_decorator


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            lines[i] = _prepare_output_docstrings(output_type, config_class)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, current docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


def is_tensor(x):
    """ Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`. """
    if is_torch_available():
        import torch

        if isinstance(x, torch.Tensor):
            return True
    if is_tf_available():
        import tensorflow as tf

        if isinstance(x, tf.Tensor):
            return True
    return isinstance(x, np.ndarray)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionnary) that will ignore the ``None`` attributes. Otherwise behaves like a
    regular python dictionary.
    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())