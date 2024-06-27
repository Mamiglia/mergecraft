import torch
from torch import nn, Tensor
from typing import List, Optional
from .base import Merger

def stock_merging(centroid: Tensor, pretrained: Tensor, N:int) -> Tensor:
    """
    Implements the stock merging from the paper: jang2024model.
    This method takes in input a handful of models and computes their average.
    Then it takes the original pretrained model as an anchor and finds the real centroid of the models.
    
    Args:
    - models (list): list of models to be merged, fine-tuned
    - base (int): the index of the base model in the list (pre trained)
    
    Returns:
    - Tensor: merged model
    """
    theta = pretrained.dot(centroid) / max(pretrained.norm() * centroid.norm(), 1e-8)
    T = N*theta / (1 + (N-1)*theta) # this is the ratio of the centroid to the anchor
    print('Ratio:', T, 'Angle between centroid and pretrained:', theta)
    return T*centroid + (1-T)*pretrained


class StockMerger(Merger):
    def __init__(self, models: List[Tensor], task:Optional[str] = None, base_index:int=0, passthrough_layers:List[str]=[], **merge_args):
        super().__init__(models, task, passthrough_layers, **merge_args)
        self.pretrained = self.tensors.pop(base_index)

    def merge(self) -> nn.Module: 
        merged_weights = stock_merging(self.baseline, self.pretrained, len(self.tensors))
        
        return self.tensor2model(merged_weights)
    
