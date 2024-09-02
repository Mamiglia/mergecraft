from typing import Iterable

from torch import Tensor
from torch.nn.functional import cosine_similarity

from mergecraft.merging.base import layer_merge

@layer_merge
def stock(models: Iterable[Tensor], base_index:int=0, **_) -> Tensor:
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
    theta = cosine_similarity(centroid.flatten(), pretrained.flatten(), dim=0).clamp(min=0, max=1)
    T = N*theta / (1 + (N-1)*theta) # this is the ratio of the centroid to the anchor

    return T*centroid + (1-T)*pretrained
