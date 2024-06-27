import torch
from torch import nn, Tensor
from typing import Iterable
from .base import Merger
import numpy as np

def dare_merging(models: Iterable[Tensor], pretrained: Tensor, p: float = 0.1) -> Tensor:
    """
    Implements the model merging algorithm from the paper: 
        "Language Models are Super Mario: Absorbing abilities from homologous models as a Free Lunch"
    
    The Drop And REscale method randomly drops some elements of the task vectors and rescales back

    Args:
    - models (Iterable[Tensor]): list of models to be merged, fine-tuned
    - pretrained (Tensor): the base model to be merged, pre-trained
    - p (float): the probability of dropping an element

    Returns:
    - Tensor: merged model
    """
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
    merged_task /= drop_mask.logical_not().sum(dim=0)
    merged_task[torch.isnan(merged_task)] = 0
    return pretrained + merged_task

class DAREMerger(Merger):
    def __init__(self, models: Iterable[nn.Module], base_index:int=0, p: int = 0.5, **merge_args):
        super().__init__(models, **merge_args)
        self.p = p
        self.pretrained = self.tensors.pop(base_index)
    
    def merge(self) -> nn.Module:
        merged_weights = dare_merging(self.tensors, self.pretrained, self.p)
        return self.tensor2model(merged_weights)