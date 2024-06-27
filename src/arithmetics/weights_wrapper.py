from copy import deepcopy
import torch
from torch import nn
from collections import OrderedDict
from typing import Dict, Callable, List, Any, Optional#,Self # need python 3.11
from numbers import Number
from torch import Tensor

class ArchitectureTensor:
    '''Defines a map between the model's parameters and a tensor. 
        It takes in input a torch model and defines a mapping between the model's parameters and a tensor.
        In particular this mapping is valid also for any other model with the same architecture.
        The tensor can be converted back to a model with the same architecture.
        The tensor can be used to store the model's parameters and can be used to perform operations on the model's parameters.
        
        Example:
            model_a = nn.Linear(10, 5)
            model_b = nn.Linear(10, 5)
            arch2tensor = ArchitectureTensor(model_a)
            
            tensor_a = arch2tensor.to_tensor(model_a)
            tensor_b = arch2tensor.to_tensor(model_b)
            tensor_c = tensor_a + tensor_b
            
            model_c = arch2tensor.to_model(tensor_c)
    '''
    def __init__(self, model:nn.Module) -> None:
        self.arch = model
        self.layers_shape = [p.shape for p in model.parameters()]
        self.total_size = sum(p.numel() for p in model.parameters())
        self.device = next(model.parameters()).device
        
    def to_tensor(self, params: nn.Module | List[torch.Tensor]) -> Tensor:
        '''Converts the model to a tensor'''
        if isinstance(params, nn.Module):
            return self.to_tensor(params.parameters())
        data = torch.cat([p.flatten() for p in params])
        return data.clone().detach()

    def to_model(self, tensor: Optional[Tensor]= None):
        '''Converts the tensor rappresenting an architecture to a model'''
        assert tensor.numel() == self.total_size, f'The tensor size {tensor.numel()} does not match the model size {self.total_size}'
 
        i = 0
        new_model = deepcopy(self.arch)
        for name, param in new_model.named_parameters():
            size = param.numel()
            param.data = tensor[i:i+size].reshape(param.shape)
            i += size
            
        return new_model
    
    def zeros(self):
        '''Returns a tensor of zeros with the same size as the model'''
        return torch.zeros(self.total_size, dtype=torch.float32, device=self.device, requires_grad=False)
    
    def name_mask(self, names: List[str]):
        '''Returns a mask of the parameters that have the given names'''
        named = [torch.ones_like(p, dtype=bool, device=self.device).flatten() & (name in names) for name,p in self.arch.named_parameters()]
        return torch.cat(named)
    
    def grad_mask(self):
        '''Returns a mask of the parameters that require gradients'''
        require_grads = [torch.ones(p.size(), dtype=bool, device=self.device).flatten() & p.requires_grad for p in self.arch.parameters()]
        return torch.cat(require_grads)
        
    

# class ModelArithmetics(nn.Module):
#     def __init__(self, model: nn.Module):
#         super().__init__()
#         self.model = deepcopy(model)
        
#         self.names, self.weights = list(zip(*self.model.named_parameters()))

#     @property
#     def named_parameters(self):
#         return OrderedDict(zip(self.names, self.weights))
    
#     def __add__(self, other):
#         return self.__pair_op__(other, torch.add)
    
#     def __sub__(self, other):
#         return self.__pair_op__(other, torch.sub)
    
#     def __mul__(self, other):
#         if isinstance(other, Number) or isinstance(other, torch.Tensor):
#             return self.__scalar_op__(other, torch.mul)
#         return self.__pair_op__(other, torch.mul)
    
#     def __truediv__(self, other):
#         if isinstance(other, Number) or isinstance(other, torch.Tensor):
#             return self.__scalar_op__(other, torch.div)
#         return self.__pair_op__(other, torch.div)
    
#     def __pair_op__(self, other : List[torch.Tensor] | nn.Module | Any, op: Callable[[Tensor, Tensor], Tensor]):
#         if isinstance(other, ModelArithmetics):
#             other = other.weights
#         if isinstance(other, nn.Module):
#             other = list(other.parameters())
#         self.weights = [op(s, o) for s, o in zip(self.weights, other)]
#         return self
    
#     def __scalar_op__(self, scalar: Number | Tensor, op: Callable[[Tensor, Any], Tensor]):
#         self.weights = [op(s, scalar) for s in self.weights]
#         return self

    # def __iadd__(self, other):
    #     return self.__i_pair_op__(other, torch.add)
    
    # def __isub__(self, other):
    #     return self.__i_pair_op__(other, torch.sub)
    
    # def __imul__(self, other):
    #     if isinstance(other, Number) or isinstance(other, torch.Tensor):
    #         return self.__i_scalar_op__(other, torch.mul)
    #     return self.__i_pair_op__(other, torch.mul)
    
    # def __itruediv__(self, other):
    #     if isinstance(other, Number) or isinstance(other, torch.Tensor):
    #         return self.__i_scalar_op__(other, torch.div)
    #     return self.__i_pair_op__(other, torch.div)
    
    # def __eq__(self, other):
    #     '''Computes a element-wise comparison between the weights of two models. returning a mask'''
    #     mask = self.copy()
    #     return mask.__pair_op__(other, torch.eq)
    
    # def to_model(self):
    #     return self.model
    
    # def copy(self):
    #     return deepcopy(self)
    
    # @staticmethod
    # def zeros_like(model: nn.Module):
    #     return ModelArithmetics(model) * 0


    
    