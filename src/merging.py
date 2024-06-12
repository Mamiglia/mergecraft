import torch
from torch import nn
from src.weights_wrapper import ArchitectureTensor
from typing import Iterable

def weighted_merging(models: Iterable[nn.Module], weights: Iterable[float | torch.Tensor]):
    """
    Merges the models with the given weights.
    
    Args:
    - models (list): list of models to be merged
    - weights (list): list of weights to be used for merging
    
    Returns:
    - nn.Module: merged model
    """
    arch = ArchitectureTensor(next(iter(models)))
    
    params = [arch.to_tensor(model) for model in models]
    merged_weights = sum(w * m for w, m in zip(weights, params))
    merged_weights /= sum(weights)
    
    if torch.isnan(merged_weights).any():
        # handle NaN values
        print('NaNs in the merged weights')
        soup = sum(params) / len(params)
        merged_weights[torch.isnan(merged_weights)] = soup[torch.isnan(merged_weights)]
        
    # handle non-grad values
    requires_grad = arch.grad_mask()
    if not requires_grad.all():
        merged_weights[~requires_grad] = sum(params)[~requires_grad] / len(params)
    
    return arch.to_model(merged_weights)
     
    