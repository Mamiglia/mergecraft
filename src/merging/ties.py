import torch
from torch import nn, Tensor
from src.arithmetics.weights_wrapper import ArchitectureTensor
from typing import Iterable
from .base import Merger
import numpy as np

def ties_merging(models: Iterable[Tensor], pretrained: Tensor, k: float = 0.1, full_quantile_computation: bool = False) -> Tensor:
    """
    Implements the model merging algorithm from the paper:

    Args:
    - models (Iterable[Tensor]): list of models to be merged, fine-tuned
    - pretrained (Tensor): the base model to be merged, pre-trained
    - k (float): the trimming factor, top-k% of the weights are kept
    - full_quantile_computation (bool): flag to perform full quantile computation or not

    Returns:
    - Tensor: merged model
    """
    tasks = torch.stack([model - pretrained for model in models])  # list of task vectors
    del models
    
    # trim: keep only top-k% magnitude of the weights
    magnitudes = tasks.numpy().astype(np.half)
    if not full_quantile_computation:
        # takes a subset to ease computation
        subset = int((1.96*np.sqrt(k*(1-k))/0.005)**2)
        magnitudes = np.random.choice(magnitudes.flat, subset, replace=False)
        print('Subset:', subset)
    magnitudes = np.abs(magnitudes)
    threshold = np.quantile(magnitudes, 1 - k)
    del magnitudes
    print(threshold)
    is_topk = tasks.abs() >= threshold
    print('Selected:', is_topk.to(float).mean())
    tasks[~is_topk] = 0

    # elect: for each parameter find the sign with the highest total magnitude
    signs = tasks.sum(dim=0).sign()
    agrees_elect_sign = tasks.sign() == signs.unsqueeze(0)
    print(agrees_elect_sign.shape)
    del signs
    tasks[~agrees_elect_sign] = 0

    # disjoint merge
    merged_task = tasks.sum(dim=0) 
    del tasks
    selected = (agrees_elect_sign & is_topk).sum(dim=0)
    merged_task /= selected
    merged_task[selected == 0] = 0
    return pretrained + merged_task


class TIESMerger(Merger):
    def __init__(self, models: Iterable[nn.Module], base_index:int=0, k: int = 0.1, **merge_args):
        super().__init__(models, **merge_args)
        self.k = k
        self.pretrained = self.tensors.pop(base_index)

    def merge(self) -> nn.Module:
        merged_weights = ties_merging(self.tensors, self.pretrained, self.k)
        return self.tensor2model(merged_weights)
        