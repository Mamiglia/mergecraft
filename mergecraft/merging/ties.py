import torch
from torch import nn, Tensor
from typing import List, Optional, Iterable
from .base import model_merge
from mergecraft.arithmetics.weights_wrapper import StateDict, dict_map
import numpy as np

@dict_map
def ties_layer_merging(models: Iterable[Tensor], base_index:int=0, k: float = 0.1) -> Tensor:
    """
    Implements the model merging algorithm from the paper:

    Args:
    - models (Iterable[Tensor]): list of models to be merged, fine-tuned
    - pretrained (Tensor): the base model to be merged, pre-trained
    - k (float): the trimming factor, top-k% of the weights are kept
    - full_quantile_computation (bool): flag to perform full quantile computation or not

    Returns:
    - Tensor: merged model
    """
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

@model_merge
def ties(models: Iterable[StateDict], base_index:int=0, k: float = 0.1, **kwargs):
    '''Merges the models by taking the mean of the weights'''
    
    return ties_layer_merging(models, base_index, k)
