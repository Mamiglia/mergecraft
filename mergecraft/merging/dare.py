import torch
from torch import nn, Tensor
from typing import List, Optional, Iterable
from .base import model_merge
from src.arithmetics.weights_wrapper import StateDict, dict_map

@dict_map
def dare_layer_merging(models: Iterable[Tensor], base_index:int=0, p: float = 0.5) -> Tensor:
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

@model_merge
def dare(models: Iterable[StateDict], base_index:int=0, p: float = 0.5, **kwargs) -> StateDict:
    '''Merges the models by applying the DARE procedure as introduced in the paper:
        "Language Models are Super Mario: Absorbing abilities from homologous models as a Free Lunch" from Yu et al.
        The DARE procedure randomly drops some elements of the task vectors and rescales back
        
    Args:
    - models (Iterable[StateDict]): list of models to be merged, fine-tuned and the original pre-trained model
    - base_index (int): the index of the base model
    - p (float): the percentage of elements to be dropped
    
    Returns:
    - StateDict: merged model
    '''
    return dare_layer_merging(models, base_index, p)
