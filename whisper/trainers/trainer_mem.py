
from transformers import Seq2SeqTrainer, Trainer
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers.utils import logging

from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)


logger = logging.get_logger(__name__)

if is_datasets_available():
    import datasets

'''
for transformers-4.37.1
and 
transformers==4.38.2
'''
class MemSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(self, eval_data_collator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator if eval_data_collator is not None else self.data_collator

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        try:
            r = torch.cuda.memory_reserved(self.args.device)
            if r > 42574282752: torch.cuda.empty_cache()
            return super().training_step(model, inputs)
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory. Handling the error...", flush=True)
            torch.cuda.empty_cache()
            return torch.tensor(float('inf'), requires_grad=True, device=self.args.device)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.eval_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

       	dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
       	}
        try:
            if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        except Exception as e:
            if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
                dataloader_params["drop_last"] = self.args.dataloader_drop_last            	
	

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
