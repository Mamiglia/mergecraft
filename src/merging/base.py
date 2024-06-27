import torch
from torch import nn, Tensor
from functools import cached_property
from src.arithmetics.weights_wrapper import ArchitectureTensor
from typing import Iterable, Optional
from transformers import pipeline 

def weighted_merging(models: Iterable[Tensor], weights: Optional[Iterable[float | Tensor]] = None, eps:float=1e-8)-> Tensor:
    """
    Merges the models with the given weights.
    
    Args:
    - models (list): list of models to be merged
    - weights (list): list of weights to be used for merging, if not given will merge with equal weights
    
    Returns:
    - Tensor: merged model
    """
    if weights is None:
        weights = torch.ones(len(models), dtype=torch.float32)
    else:
        weights = torch.tensor(weights, dtype=torch.float32)
    
    merged_weights = sum(w * m for w, m in zip(weights, models))
    merged_weights /= torch.clamp(sum(weights), min=eps)
    
    return merged_weights

class Merger:
    '''Generic Merger abstract class'''
    def __init__(self, 
                 models:Iterable[str | nn.Module], 
                 task:Optional[str] = None, 
                 passthrough_layers:Iterable[str]=[], 
                 **merge_args):
        self.load_models(models, task) # stores architecture, tensors and pipes
        self.passthrough_layers = passthrough_layers
        
    @cached_property
    def baseline(self) -> torch.Tensor:
        '''Returns the baseline model, the mean of the models to be merged. 
            This is used as a reference point for merging, when some values cannot be merged normally'''
        return weighted_merging(self.tensors)

    def load_models(self, models:Iterable[str | nn.Module], task:Optional[str] = None):
        if isinstance(models[0], str):
            assert task is not None, 'Task must be provided when models are huggingface pipelines'
            self.pipes = [self.hf_load(model, task) for model in models]
            models = [pipe.model for pipe in self.pipes]
        else: 
            self.pipes = models

        self.arch = ArchitectureTensor(models[0])
        self.tensors = [self.arch.to_tensor(model) for model in models]

    def tensor2model(self, merged_weights:torch.Tensor) -> nn.Module:
        # handle non-grad values, not copying them
        requires_grad = self.arch.grad_mask()
        is_nan = torch.isnan(merged_weights)
        avoid_layers = self.arch.name_mask(self.passthrough_layers)
        passthrough_mask = ~requires_grad | is_nan | avoid_layers

        if passthrough_mask.any():
            print(f'Passing through {passthrough_mask.sum()} weights, of which: {is_nan.sum()} are NaN and {(~requires_grad).sum()} don\'t require gradient.')
            merged_weights[passthrough_mask] = self.baseline[passthrough_mask]

        return self.arch.to_model(merged_weights)
    
    @staticmethod
    def hf_load(model:str, task:str):
        '''Load a huggingface pipeline'''
        return pipeline(task, model=model, device='cpu', framework='pt')
    
    def merge(self):
        raise NotImplementedError('This method should be implemented in the child class')


class WeightedMerger(Merger):
    def __init__(self, models:Iterable[Tensor], weights: Optional[Iterable[float | Tensor]] = None, **merge_args):
        super().__init__(models, **merge_args)
        self.weights = weights

    def merge(self) -> nn.Module:
        merged_weights = weighted_merging(self.tensors, self.weights)
        return self.tensor2model(merged_weights)