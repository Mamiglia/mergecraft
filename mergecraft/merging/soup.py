from typing import Iterable, Optional

import torch
from torch import Tensor

from mergecraft.merging.base import model_merge
from mergecraft.arithmetics.weights_wrapper import StateDict

@model_merge
def soup(models: Iterable[StateDict], weights: Optional[Iterable[float]] = None, **_)-> Tensor:
    """
    Merges the models with the given weights.
    
    Args:
    - models (list): list of models to be merged
    - weights (list): list of weights to be used for merging, if not given will merge with equal weights
    
    Returns:
    - Tensor: merged model
    """
    if weights is None:
        weights = [1] * len(models)
    assert len(weights) == len(models), 'The number of weights must match the number of models'
        
    merged_weights = sum(m*w for w, m in zip(weights, models))
    merged_weights /= sum(weights)
    
    return merged_weights
