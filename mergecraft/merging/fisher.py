import torch
from torch import nn, Tensor
from functools import cached_property
from typing import Iterable, Optional, Any
from types import MethodType
from transformers import pipeline, Pipeline
from tqdm import tqdm
import time
import gc
from src.arithmetics.weights_wrapper import StateDict, dict_map
from .base import model_merge
from ..computation.hf_extension import HessianCallback, add_callback

@dict_map(names=['models', 'hessians'])
def fisher_layer_merging(models: Iterable[Tensor], hessians: Iterable[Tensor], eps:float=1e-8) -> Tensor:
    '''Merges the weights of the models based on the Fisher Information Matrix diagonal approximation.
    
    Args:
        weights (Iterable[Tensor]): the weights of the models
        hessians (Iterable[Tensor]): the Fisher Information Matrix diagonal approximation of the models
    Returns:
        Tensor: the merged weights
    '''
    models = torch.stack(models)
    hessians = torch.stack(hessians)
    return (models * hessians).sum(dim=0) / hessians.sum(dim=0).clamp(min=eps)

@model_merge
def fisher(
        models: Iterable[StateDict],
        pipelines: Iterable[Pipeline],
        dataset, 
        **merge_args
        ) -> StateDict:
    '''Merges the weights of the models based on the Fisher Information Matrix diagonal approximation.
    
    Args:
    Returns:
        StateDict: the merged weights
    '''
    hessians = [estimateFIMDiagonal(pipe, dataset) for pipe in tqdm(pipelines)]
        
    return fisher_layer_merging(models, hessians)


def estimateFIMDiagonal(pipe:Pipeline, dataset: Any) -> StateDict:
    """Computes the Fisher Information Matrix diagonal approximation for the given dataset and model pipeline.
        Args:
            pipe (Pipeline): the pipeline model
            dataset (Any): the dataset to compute the Fisher Information Matrix for
        Returns:
            List[torch.Tensor]: the Fisher Information Matrix diagonal approximation
    """
    model = pipe.model
    # initialize the callback
    hess_callback = HessianCallback(model)

    # Modify the pipeline to compute the hessian
    pipe.get_inference_context = lambda *_: torch.enable_grad   # enable gradient computation
    pipe._forward = MethodType(add_callback(pipe._forward, hess_callback), pipe) # add the callback to the forward method

    predicted = list(pipe(dataset, batch_size=1))
    
    hessian = hess_callback.get_hessian().to('cpu')

    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    return hessian
