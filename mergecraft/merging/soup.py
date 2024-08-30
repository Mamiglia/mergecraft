from copy import deepcopy
import torch
from torch import nn, Tensor
from mergecraft.arithmetics.weights_wrapper import StateDict, dict_map
from typing import Iterable, Optional
from mergecraft.merging.base import model_merge

@dict_map
def weighted_layer_merging(models: Iterable[Tensor], weights: Optional[Iterable[float | Tensor]] = None, eps:float=1e-8)-> Tensor:
    """
    Merges the models with the given weights.
    
    Args:
    - models (list): list of models to be merged
    - weights (list): list of weights to be used for merging, if not given will merge with equal weights
    
    Returns:
    - Tensor: merged model
    """
    if weights is None:
        weights = [torch.ones(1)] * len(models)
    else:
        assert len(weights) == len(models), 'The number of weights must match the number of models'
        weights = [torch.tensor(w) if not isinstance(w, Tensor) else w for w in weights]
        
    merged_weights = sum(w * m for w, m in zip(weights, models))
    merged_weights /= torch.clamp(sum(weights), min=eps)
    
    return merged_weights


@model_merge
def soup(
        models: Iterable[StateDict], 
        weights : Optional[Iterable[float|Tensor]] = None, 
        **kwargs) -> StateDict:
    '''Merges the models by taking the mean of the weights'''
    return weighted_layer_merging(models, weights)