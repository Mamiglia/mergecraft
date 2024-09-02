from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from mergecraft.merging.base import layer_merge

@layer_merge
def ties(models: Iterable[Tensor], base_index:int=0, k: float = 0.1, **_) -> Tensor:
    '''Implements the model merging algorithm from the paper:
        "TIES-Merging: Resolving Interference When Merging Models" by Yadav et al. 2023

    Args:
    - models (Iterable): list of models to be merged, fine-tuned and baseline
    - base_index (int): index of the baseline model which was the base for fine-tunin
    - k (float): trimming factor, the top-k% of the weights that are going to be kept

    Returns:
    - Merged model
    '''
    pretrained = models.pop(base_index)
    tasks = torch.stack([model - pretrained for model in models])  # list of task vectors
    del models
    
    # trim: keep only top-k% magnitude of the weights
    magnitudes = np.abs(tasks.numpy(force=True).astype(np.half))
    threshold = np.quantile(magnitudes.flat, 1 - k)
    del magnitudes
    is_topk = tasks.abs() >= threshold
    tasks[~is_topk] = 0

    # elect: for each parameter find the sign with the highest total magnitude
    signs = tasks.sum(dim=0).sign()
    agrees_elect_sign = tasks.sign() == signs.unsqueeze(0)
    del signs
    tasks[~agrees_elect_sign] = 0

    # disjoint merge
    merged_task = tasks.sum(dim=0) 
    del tasks
    selected = (agrees_elect_sign & is_topk).sum(dim=0)
    merged_task /= selected
    merged_task[selected == 0] = 0
    return pretrained + merged_task
