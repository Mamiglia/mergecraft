import torch
from torch import nn, Tensor
from typing import Iterable, Optional, List
from .base import Merger


def task_merging(models: Iterable[Tensor], pretrained: Tensor, signs: Optional[List[int]]=None) -> Tensor:
    """
    Implements the model merging algorithm from the paper: 
        "Editing Models with Task Arithmetic", Ilharco et al. 2023
    
    Eve

    Args:
    - models (Iterable[Tensor]): list of models to be merged, fine-tuned
    - pretrained (Tensor): the base model to be merged, pre-trained
    - k (int): the trimming factor, top-k% of the weights are kept

    Returns:
    - Tensor: merged model
    """
    if signs is None:
        signs = [1] * len(models)
    assert len(signs) == len(models), 'Number of signs should be equal to the number of models'
    assert all(s in [-1, 1] for s in signs), 'Sign should be either -1 or 1'
        
    tasks = [model - pretrained for model in models]  # list of task vectors
    tasks = [task * s for task, s in zip(tasks, signs)] 
    merged_task = sum(tasks) / len(tasks)
    return pretrained + merged_task

class TaskMerger(Merger):
    def __init__(self, models: Iterable[nn.Module], base_index:int=0, signs:Optional[List[int]]=None, **merge_args):
        super().__init__(models, **merge_args)
        self.pretrained = self.tensors.pop(base_index)
        self.signs = signs
    
    def merge(self, signs:Optional[List[int]]= None) -> nn.Module:
        if signs is None:
            signs = self.signs
        merged_weights = task_merging(self.tensors, self.pretrained, signs=signs)
        return self.tensor2model(merged_weights)