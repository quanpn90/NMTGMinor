import logging, traceback
import os
import torch

def torch_persistent_save(*args, **kwargs):
    
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


"""
    Due to the fact that opts can change rapidly
    This function simply 
"""
def update_training_opt(opts):
    
    new_opts = opts
    
    
    
    return opts
