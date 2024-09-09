import inspect
from collections import OrderedDict
from copy import deepcopy
from functools import wraps
from numbers import Number
from typing import Any, Callable, Dict, Iterable

import torch
from torch import Tensor, nn
from transformers import pipeline


class StateDict(OrderedDict):
    '''A wrapper around a dictionary of tensors that provides arithmetic operations between the tensors.
        This class is useful to perform operations on the weights of a model.
        
    '''
    @staticmethod
    def from_model(model: nn.Module):
        return StateDict(model.named_parameters())
    
    @staticmethod
    def from_hf(name: str, **kwargs):
        from transformers import logging
        logging.set_verbosity_error()
        pipe = pipeline(model=name, framework='pt', **kwargs)
        return StateDict.from_model(pipe.model)
    
    def __add__(self, other):
        if isinstance(other, Number) or isinstance(other, torch.Tensor):
            return self.__scalar_op__(other, torch.add)
        return self.__pair_op__(other, torch.add)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Number) or isinstance(other, torch.Tensor):
            return self.__scalar_op__(other, torch.sub)
        return self.__pair_op__(other, torch.sub)
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, Number) or isinstance(other, torch.Tensor):
            return self.__scalar_op__(other, torch.mul)
        return self.__pair_op__(other, torch.mul)
    
    def __truediv__(self, other):
        if isinstance(other, Number) or isinstance(other, torch.Tensor):
            return self.__scalar_op__(other, torch.div)
        return self.__pair_op__(other, torch.div)
    
    def __pair_op__(self, other : Dict[str, Tensor] | nn.Module, op: Callable[[Tensor, Tensor], Tensor], strict:bool=False):
        if isinstance(other, nn.Module):
            other = StateDict.from_model(other)
        assert isinstance(other, StateDict), 'The input must be a StateDict or a nn.Module'
        common_keys = self.keys() & other.keys()
        if strict:
            assert len(common_keys) == len(self.keys()), 'The two models must have the same parameters'
        
        return StateDict({k: op(self[k], other[k]) for k in common_keys})

    def __scalar_op__(self, scalar: Number | Tensor, op: Callable[[Tensor, Any], Tensor]):
        r = {k: op(v, scalar) for k,v in self.items()}
        return StateDict(r)
    
    def __i_pair_op__(self, other : Dict[str, Tensor] | nn.Module, op: Callable[[Tensor, Tensor], Tensor], strict:bool=False):
        if isinstance(other, nn.Module):
            other = StateDict.from_model(other)
        assert isinstance(other, StateDict), 'The input must be a StateDict or a nn.Module'
        common_keys = self.keys() & other.keys()
        if strict:
            assert len(common_keys) == len(self.keys()), 'The two models must have the same parameters'
        
        self.update({k:op(self[k], other[k]) for k in common_keys})
        return self
    
    def __i_scalar_op__(self, scalar: Number | Tensor, op: Callable[[Tensor, Any], Tensor]):
        self.update({k:op(v, scalar) for k,v in self.items()})
        return self                 

    def __iadd__(self, other):
        return self.__i_pair_op__(other, torch.add)
    
    def __isub__(self, other):
        return self.__i_pair_op__(other, torch.sub)
    
    def __imul__(self, other):
        if isinstance(other, Number) or isinstance(other, torch.Tensor):
            return self.__i_scalar_op__(other, torch.mul)
        return self.__i_pair_op__(other, torch.mul)
    
    def __itruediv__(self, other):
        if isinstance(other, Number) or isinstance(other, torch.Tensor):
            return self.__i_scalar_op__(other, torch.div)
        return self.__i_pair_op__(other, torch.div)
    
    def to(self, device):
        return StateDict({k: v.to(device) for k,v in self.items()})
    
    def filter_values(self, condition: Callable[[Tensor], bool]):
        filtered = [k for k,v in self.items() if not condition(v)]
        return self.remove(filtered)
        
    def remove(self, keys: Iterable[str]):
        for k in keys:
            del self[k]
        return self
    
    def subset(self, keys: Iterable[str]):
        return StateDict({k: self[k] for k in keys})
    
    def copy(self):
        return deepcopy(self)
    
    def to_model(self, model:str|nn.Module|dict, verbose:bool=False, **kwargs) -> str|nn.Module|dict:
        '''Converts a state_dict to a model'''
        if isinstance(model, str):
            task = kwargs.get('task')
            pipe = pipeline(task, model, framework='pt', device='cpu')
            missing_keys = pipe.model.load_state_dict(self, strict=False)
            if verbose:
                print('Missing keys:', missing_keys)
            return pipe
        if isinstance(model, nn.Module):
            model = deepcopy(model)
            model.load_state_dict(self, strict=False)
            return model
        if isinstance(model, dict):
            return self
        raise ValueError('Model must be either a path to a huggingface model, a nn.Module or a dictionary')
    
    @staticmethod
    def zeros_like(item, **kwargs):
        if isinstance(item, nn.Module):
            return StateDict.from_model(item) * 0
        if isinstance(item, StateDict):
            return item * 0
        if isinstance(item, dict):
            return StateDict(item) * 0
        if isinstance(item, str):
            return StateDict.from_hf(item, **kwargs) * 0
        raise ValueError('The input must be a nn.Module, a StateDict, a dictionary or a string')
    
    

def dict_map(func_or_names=None):
    '''Decorator that applies a function to the tensor values of a dictionary
    
    Can be used as:
    @dict_map
    or
    @dict_map(names=['param1', 'param2'])
        
    Args:
        func_or_names: Either the function being decorated or the names of 
                       the dictionaries to apply the function to.
                       If None or not specified, the function is applied to 
                       the first argument.
    '''
    if callable(func_or_names):
        # The decorator was used without arguments: @dict_map
        func = func_or_names
        names = None
        return _dict_map_wrapper(func, names)
    else:
        # The decorator was used with arguments: @dict_map(names=...)
        names = func_or_names
        return lambda func: _dict_map_wrapper(func, names)


def _dict_map_wrapper(func, names):
    # Identify the parameters of the function which are dictionaries
    params = list(inspect.signature(func).parameters.keys())

    if names is None:
        names = params[:1]
    
    assert all(name in params for name in names), 'The parameter names must be part of the function signature'
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> StateDict:
        kwargs.update(zip(params, args))
        dict_args = {}
        for name in names:
            assert name in kwargs, f'The list of dictionaries "{name}" is missing'
            dict_args[name] = kwargs.pop(name)

            assert isinstance(dict_args[name], list), f'The argument {name} must be a list of dictionaries'
            assert all(len(d) == len(dict_args[names[0]]) for d in dict_args.values()), 'The list of dictionaries must have the same length'
            assert all(isinstance(d, dict) for d in dict_args[name]), f'The argument {name} must be a list of dictionaries'

        # Find common keys in all dictionaries
        common_keys = set.intersection(*[set(d.keys()) for arg in dict_args.values() for d in arg ])
        assert len(common_keys) > 0, 'The input dictionaries must have at least one common key'

        with torch.no_grad():
            res = {k: func(**{name: [d[k] for d in dict_args[name]] for name in names}, layer_name=k, **kwargs) for k in common_keys}
        return StateDict(res)
    
    return wrapper    
