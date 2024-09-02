from typing import Iterable

from torch.nn.functional import cosine_similarity
from torch import Tensor
import torch

from mergecraft.merging.base import layer_merge

@layer_merge
def slerp(
    models: Iterable[Tensor],
    tau: float = 0.5, eps:float=1e-7, **kw
    ) -> Tensor:
    """
    Merges the models using the Spherical Linear Interpolation (SLERP) method.
    
    Args:
    - models (list): The two models to be merged
    - tau (float): The interpolation factor
    
    Returns:
    - Tensor: merged model
    """
    assert 0 <= tau <= 1, 'Tau should be between 0 and 1'
    assert len(models) == 2, 'SLERP can only be applied to two models'
    model1, model2 = models
    
    cos_theta = cosine_similarity(model1.view(-1), model2.view(-1), dim=0)
    
    if eps < cos_theta < 1 - eps:
        theta = torch.acos(cos_theta)
        lambda1 = torch.sin((1 - tau) * theta) / torch.sin(theta)
        lambda2 = torch.sin(tau * theta) / torch.sin(theta)
    else:
        if 'verbose' in kw and kw['verbose']:
            print(kw['layer_name'], 'is collinear')
        lambda1 = 1 - tau
        lambda2 = tau

    return lambda1 * model1 + lambda2 * model2