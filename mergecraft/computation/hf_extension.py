import torch
from torch import nn
from transformers import Pipeline
from transformers.utils.generic import ModelOutput

from mergecraft.arithmetics.weights_wrapper import StateDict
from mergecraft.computation.hessian import hess


class HessianCallback:
    hessian: StateDict
    num_samples: int
    
    def __init__(self, model):
        self.model = model
        self.hessian = StateDict.zeros_like(model)
        self.num_samples = 0

    def __call__(self, outputs: ModelOutput):
        # can probaly be improved with functorch
        H = hess(self.model, outputs.logits)
        self.num_samples += 1
        with torch.no_grad():
            self.hessian += StateDict(H)
        
        # detach as hf needs clean logits with no gradient
        outputs.logits = outputs.logits.detach()
    
    @torch.no_grad()
    def get_hessian(self) -> StateDict:
        return self.hessian / self.num_samples
    
    
def add_callback(func, callback):
    '''This functions injects a callback into a method of a class.
    
    Args:
        func (function): the method to inject the callback into
        callback (function): the callback to inject
    Returns:    
        function: the new method with the callback injected
    '''
    def new_func(self, *args, **kwargs):
        outputs = func.__func__(self, *args, **kwargs)
        callback(outputs)
        return outputs
    return new_func
