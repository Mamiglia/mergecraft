from typing import Iterable

import torch
from torch import Tensor

from mergecraft.merging.base import layer_merge

@layer_merge
def dare(models: Iterable[Tensor], base_index:int=0, p: float = 0.5, **_) -> Tensor:
    """
    Implements the merging procedure DARE for the single layer level.
    
    Args:
    - models (Iterable[Tensor]): list of models to be merged, fine-tuned and the original pre-trained model
    - base_index (int): the index of the base model
    - p (float): the probability of dropping an element

    Returns:
    - Tensor: merged model
    """
    pretrained = models.pop(base_index)
    tasks = torch.stack([model - pretrained for model in models])  # list of task vectors
    del models    
    # randomly drop p% of the elements
    drop_mask = torch.rand(tasks.shape) < p
    tasks[drop_mask] = 0
    # rescale back
    tasks /= (1-p)
    # merge
    merged_task = tasks.sum(dim=0) 
    del tasks
    merged_num = drop_mask.logical_not().sum(dim=0)
    merged_num[merged_num == 0] = 1
    merged_task /= merged_num
    return pretrained + merged_task
