import torch
from torch import nn
from collections import OrderedDict
from typing import Dict, Callable, List, Any#,Self # need python 3.11
import inspect
import numbers

class Weights:
    def __init__(self, parameters : Dict[str, torch.Tensor] | List[torch.Tensor], names : List[str] = None):
        if inspect.isgenerator(parameters):
            parameters = list(parameters)
            
        if isinstance(parameters[0], tuple):
            self.names = [p[0] for p in parameters] if names is None else names
            self.weights = [p[1] for p in parameters]
        else:
            self.names = names if names is not None else [f'param_{i}' for i,_ in enumerate(parameters)]
            self.weights = parameters 

    @property
    def named_parameters(self):
        return OrderedDict(zip(self.names, self.weights))
    
    def __eq__(self, other):
        return all([torch.equal(s, o) for s, o in zip(self.weights, other.weights)])
    
    def __pair_op__(self, other, op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        if not isinstance(other, Weights):
            other = Weights(other, self.names)
        return Weights([op(s, o) for s, o in zip(self.weights, other.weights)], self.names)
    
    def __scalar_op__(self, scalar, op: Callable[[torch.Tensor, Any], torch.Tensor]):
        return Weights([op(s, scalar) for s in self.weights], self.names)
    
    def __add__(self, other):
        return self.__pair_op__(other, torch.add)
    
    def __sub__(self, other):
        return self.__pair_op__(other, torch.sub)
    
    def __mul__(self, other):
        if not isinstance(other, numbers.Number) and not isinstance(other, torch.Tensor):
            return self.__pair_op__(other, torch.mul)
        return self.__scalar_op__(other, torch.mul)
    
    def __truediv__(self, other):
        if not isinstance(other, numbers.Number) and not isinstance(other, torch.Tensor):
            return self.__pair_op__(other, torch.div)
        return self.__scalar_op__(other, torch.div)
    
    def __i_pair_op__(self, other : List[torch.Tensor], op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        if isinstance(other, Weights):
            other = other.weights
        self.weights = [op(s, o) for s, o in zip(self.weights, other)]
        return self
    
    def __i_scalar_op__(self, scalar, op: Callable[[torch.Tensor, Any], torch.Tensor]):
        self.weights = [op(s, scalar) for s in self.weights]
        return self

    def __iadd__(self, other):
        return self.__i_pair_op__(other, torch.add)
    
    def __isub__(self, other):
        return self.__i_pair_op__(other, torch.sub)
    
    def __imul__(self, other):
        if not isinstance(other, numbers.Number) and not isinstance(other, torch.Tensor):
            return self.__i_pair_op__(other, torch.mul)
        return self.__i_scalar_op__(other, torch.mul)
    
    def __itruediv__(self, other):
        if not isinstance(other, numbers.Number) and not isinstance(other, torch.Tensor):
            return self.__i_pair_op__(other, torch.div)
        return self.__i_scalar_op__(other, torch.div)
    
    def to(self, arg):
        return self.__i_scalar_op__(arg, lambda x, y: x.to(y))
    
    @staticmethod
    def from_model(model: nn.Module, zero_init: bool = False):
        named_parameters = model.named_parameters()
        if zero_init:
            named_parameters = [(name,torch.zeros_like(param, requires_grad=False)) for name,param in named_parameters]
        return Weights(named_parameters)
    

    
    