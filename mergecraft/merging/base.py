from copy import deepcopy
from functools import wraps
from typing import Callable, Iterable

from torch import nn
from transformers import pipeline

from mergecraft.arithmetics.weights_wrapper import StateDict, dict_map


def model_merge(func: Callable[[Iterable[StateDict]], StateDict]) -> Callable:
    @wraps(func)
    def wrapper(models: Iterable[str | nn.Module | dict],
                passthrough_layers:Iterable[str] = (), verbose:bool = False,
                *args, **kwargs) -> nn.Module:
        if isinstance(models[-1], str):
            assert 'task' in kwargs, 'task must be provided when models are paths to huggingface models'
            state_dicts = [StateDict.from_hf(model, device='cpu', **kwargs) for model in models]
        elif isinstance(models[-1], nn.Module):
            state_dicts = [StateDict.from_model(model) for model in models]
        elif isinstance(models[-1], dict):
            state_dicts = [StateDict(model) for model in models]
        else:
            raise ValueError('Models must be either paths to huggingface models, nn.Modules or state_dicts')
        
        state_dicts = [sd.filter_values(lambda x: x.requires_grad) for sd in state_dicts]
        
        # The passthrough layers are not included in the standard merging process
        # rather for those layers, the finetuned models are averaged and integrated 
        # into the final merged weights
        passthrough_weights = {}
        if len(passthrough_layers) > 0:
            pass_sd = state_dicts
            if 'base_index' in kwargs:
                base_index = kwargs['base_index']
                pass_sd = state_dicts[:base_index] + state_dicts[base_index+1:]
            pass_sd = [sd.subset(passthrough_layers) for sd in pass_sd]
            passthrough_weights = sum(pass_sd, start=StateDict.zeros_like(pass_sd[0])) / len(pass_sd)
            
            state_dicts = [sd.remove(passthrough_layers) for sd in state_dicts]
        
        merged_weights = func(state_dicts, *args, **kwargs)
        merged_weights.update(passthrough_weights)
        
        return merged_weights.to_model(models[-1], verbose=verbose, **kwargs)
    return wrapper

def layer_merge(func: Callable[[Iterable[nn.Module]], nn.Module]) -> Callable:
    func = dict_map(func)
    func = model_merge(func)
    return func
