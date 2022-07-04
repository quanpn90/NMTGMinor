try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from omegaconf import DictConfig, OmegaConf, open_dict, _utils
import contextlib
import itertools
import logging
import re
import warnings
from typing import Optional, Tuple, Callable, Dict, List, TYPE_CHECKING
from omegaconf import DictConfig, OmegaConf, open_dict, _utils
from argparse import ArgumentError, ArgumentParser, Namespace

import numpy as np
import torch


def overwrite_args_by_name(cfg: DictConfig, overrides: Dict[str, any]):
    # this will be deprecated when we get rid of argparse and model_overrides logic

    REGISTRIES = {}

    with open_dict(cfg):
        for k in cfg.keys():
            # "k in cfg" will return false if its a "mandatory value (e.g. ???)"
            if k in cfg and isinstance(cfg[k], DictConfig):
                if k in overrides and isinstance(overrides[k], dict):
                    for ok, ov in overrides[k].items():
                        if isinstance(ov, dict) and cfg[k][ok] is not None:
                            overwrite_args_by_name(cfg[k][ok], ov)
                        else:
                            cfg[k][ok] = ov
                else:
                    overwrite_args_by_name(cfg[k], overrides)
            elif k in cfg and isinstance(cfg[k], Namespace):
                for override_key, val in overrides.items():
                    setattr(cfg[k], override_key, val)
            elif k in overrides:
                if (
                        k in REGISTRIES
                        and overrides[k] in REGISTRIES[k]["dataclass_registry"]
                ):
                    cfg[k] = DictConfig(
                        REGISTRIES[k]["dataclass_registry"][overrides[k]]
                    )
                    overwrite_args_by_name(cfg[k], overrides)
                    cfg[k]._name = overrides[k]
                else:
                    cfg[k] = overrides[k]


class omegaconf_no_object_check:
    def __init__(self):
        self.old_is_primitive = _utils.is_primitive_type

    def __enter__(self):
        _utils.is_primitive_type = lambda _: True

    def __exit__(self, type, value, traceback):
        _utils.is_primitive_type = self.old_is_primitive


def convert_namespace_to_omegaconf(args: Namespace) -> DictConfig:
    """Convert a flat argparse.Namespace to a structured DictConfig."""

    # Here we are using field values provided in args to override counterparts inside config object
    overrides, deletes = override_module_args(args)

    # configs will be in fairseq/config after installation
    config_path = os.path.join("..", "config")

    GlobalHydra.instance().clear()

    with initialize(config_path=config_path):
        try:
            composed_cfg = compose("config", overrides=overrides, strict=False)
        except:
            logger.error("Error when composing. Overrides: " + str(overrides))
            raise

        for k in deletes:
            composed_cfg[k] = None

    cfg = OmegaConf.create(
        OmegaConf.to_container(composed_cfg, resolve=True, enum_to_str=True)
    )

    # hack to be able to set Namespace in dict config. this should be removed when we update to newer
    # omegaconf version that supports object flags, or when we migrate all existing models
    from omegaconf import _utils

    with omegaconf_no_object_check():
        if cfg.task is None and getattr(args, "task", None):
            cfg.task = Namespace(**vars(args))
            from fairseq.tasks import TASK_REGISTRY

            _set_legacy_defaults(cfg.task, TASK_REGISTRY[args.task])
            cfg.task._name = args.task
        if cfg.model is None and getattr(args, "arch", None):
            cfg.model = Namespace(**vars(args))
            from fairseq.models import ARCH_MODEL_REGISTRY

            _set_legacy_defaults(cfg.model, ARCH_MODEL_REGISTRY[args.arch])
            cfg.model._name = args.arch
        if cfg.optimizer is None and getattr(args, "optimizer", None):
            cfg.optimizer = Namespace(**vars(args))
            from fairseq.optim import OPTIMIZER_REGISTRY

            _set_legacy_defaults(cfg.optimizer, OPTIMIZER_REGISTRY[args.optimizer])
            cfg.optimizer._name = args.optimizer
        if cfg.lr_scheduler is None and getattr(args, "lr_scheduler", None):
            cfg.lr_scheduler = Namespace(**vars(args))
            from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY

            _set_legacy_defaults(
                cfg.lr_scheduler, LR_SCHEDULER_REGISTRY[args.lr_scheduler]
            )
            cfg.lr_scheduler._name = args.lr_scheduler
        if cfg.criterion is None and getattr(args, "criterion", None):
            cfg.criterion = Namespace(**vars(args))
            from fairseq.criterions import CRITERION_REGISTRY

            _set_legacy_defaults(cfg.criterion, CRITERION_REGISTRY[args.criterion])
            cfg.criterion._name = args.criterion

    OmegaConf.set_struct(cfg, True)
    return cfg


def compute_mask_indices(
        shape: Tuple[int, int],
        padding_mask: Optional[torch.Tensor],
        mask_prob: float,
        mask_length: int,
        mask_type: str = "static",
        mask_other: float = 0.0,
        min_masks: int = 0,
        no_overlap: bool = False,
        min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape
    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == "xla"


def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    from .fairseq_modules import gelu, gelu_accurate

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "gelu_fast",  # deprecated
        "gelu_accurate",
        "tanh",
        "linear",
    ]
