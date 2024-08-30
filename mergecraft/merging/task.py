import torch
from torch import nn, Tensor
from typing import Iterable, Optional, List
from .base import model_merge
from mergecraft.arithmetics.weights_wrapper import StateDict, dict_map


@dict_map
def task_merging(models: Iterable[Tensor], base_index:int=0, signs: Optional[List[int]]=None) -> Tensor:
    """
    Implements the model merging algorithm from the paper: 
        "Editing Models with Task Arithmetic", Ilharco et al. 2023
    
    Args:
    - models (Iterable[Tensor]): list of models to be merged, fine-tuned
    - pretrained (Tensor): the base model to be merged, pre-trained
    - k (int): the trimming factor, top-k% of the weights are kept

    Returns:
    - Tensor: merged model
    """
    pretrained = models.pop(base_index)
        
    if signs is None:
        signs = [1] * len(models)
    assert len(signs) == len(models), 'Number of signs should be equal to the number of models'
    assert all(s in [-1, 1] for s in signs), 'Sign should be either -1 or 1'
        
    tasks = [model - pretrained for model in models]  # list of task vectors
    tasks = [task * s for task, s in zip(tasks, signs)] 
    merged_task = sum(tasks) / len(tasks)
    return pretrained + merged_task

@model_merge
def task(models: Iterable[StateDict], base_index:int=0, signs: Optional[List[int]]=None, **kwargs) -> StateDict:
    '''Merges the models by taking the mean of the weights'''
    return task_merging(models, base_index, signs)