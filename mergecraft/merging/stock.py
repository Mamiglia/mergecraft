import torch
from torch import nn, Tensor
from typing import List, Optional, Iterable
from .base import model_merge
from mergecraft.arithmetics.weights_wrapper import StateDict, dict_map


@dict_map	
def stock_layer_merging(models: Iterable[Tensor], base_index:int=0) -> Tensor:
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
    pretrained = models.pop(base_index)
    N = len(models)
    centroid = sum(models) / N
    theta = nn.functional.cosine_similarity(centroid.flatten(), pretrained.flatten(), dim=0)
    T = N*theta / (1 + (N-1)*theta) # this is the ratio of the centroid to the anchor
    return T*centroid + (1-T)*pretrained

@model_merge
def stock(models: Iterable[StateDict], base_index:int=0, **kwargs) -> StateDict:
    '''Merges the models by taking the mean of the weights'''
    return stock_layer_merging(models, base_index)
