from copy import deepcopy
import torch
from torch import nn, Tensor
from mergecraft.arithmetics.weights_wrapper import StateDict
from typing import Iterable, Optional, Callable
from transformers import pipeline 

def statedict2model(sd:StateDict, model:str|nn.Module|dict, **kwargs) -> str|nn.Module|dict:
    '''Converts a state_dict to a model'''
    if isinstance(model, str):
        task = kwargs.get('task')
        pipe = pipeline(task, model, framework='pt', device='cpu')
        missing_keys = pipe.model.load_state_dict(sd, strict=False)
        print('Missing keys:', missing_keys)
        return pipe
    if isinstance(model, nn.Module):
        model = deepcopy(model)
        model.load_state_dict(sd, strict=False)
        return model
    if isinstance(model, dict):
        return sd
    raise ValueError('Model must be either a path to a huggingface model, a nn.Module or a dictionary')
        
    
def model_merge(func: Callable[[Iterable[StateDict]], StateDict]) -> Callable:
    def wrapper(models: Iterable[str | nn.Module | dict],
                passthrough_layers:Iterable[str] = [],
                *args, **kwargs) -> nn.Module:
        if isinstance(models[-1], str):
            task = kwargs.get('task')
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
        
        return statedict2model(merged_weights, models[-1], **kwargs)
    return wrapper
