from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from dataclasses import dataclass
from typing import Dict, Union, Any
import torch
from transformers.trainer_callback import CallbackHandler
from torch import nn
from types import MethodType

@dataclass
class TrainerForwardState:
    inputs: Dict[str, Union[torch.Tensor, Any]]
    model: nn.Module
    outputs: Dict[str, Union[torch.Tensor, Any]] = None
    loss: torch.Tensor = None
    

# def on_forward_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#     print(self)
#     print(kwargs.keys())
#     forward_state = TrainerForwardState(inputs=kwargs['inputs'], model=kwargs['model'])
#     return self.call_event("on_forward_begin", args, state, control, forward_state=forward_state)

# def on_forward_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#     print(self)
#     print(kwargs.keys())
#     forward_state = TrainerForwardState(inputs=kwargs['inputs'], model=kwargs['model'], outputs=kwargs['outputs'], loss=kwargs['loss'])
#     return self.call_event("on_forward_end", args, state, control, forward_state=forward_state)

# CallbackHandler.on_forward_begin = MethodType(on_forward_begin, CallbackHandler)
# CallbackHandler.on_forward_end = MethodType(on_forward_end, CallbackHandler)

class TrainerWithForwardCallback(Trainer):
    def __init__(self, forward_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_callback = forward_callback

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override the compute_loss method to add the callback events
        """
        
        self.forward_callback.on_begin(inputs=inputs, model=model)

        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
    
        self.forward_callback.on_end(inputs=inputs, model=model, outputs=outputs, loss=loss)
        return (loss, outputs) if return_outputs else loss
    
