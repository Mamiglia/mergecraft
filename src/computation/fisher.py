from typing import Dict, List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from einops import reduce, einsum
from tqdm import tqdm

def hess(model: nn.Module, logits: torch.Tensor) -> List[torch.Tensor]:
    '''Computes the diagonal approximation of the Hessian of 
        the model output wrt to the model parameters
    
    Args:
        model (nn.Module): the model
        logits (torch.Tensor): the logits of the model output
    Returns:
        List[torch.Tensor]: the Hessian of the model output wrt to the model parameters
    '''
    logits = logits.squeeze(0)
    assert logits.dim() == 1, "logits should be 1D"
    probs     = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    grads = [compute_grad(model, log_probs[i], probs[i]) for i in range(logits.shape[0])]
    
    grads = zip(*grads) # pair up the layers
    grads = [torch.stack(g).sum(0) for g in grads]

    return grads

@torch.no_grad()
def compute_grad(model: nn.Module, log_probs: torch.Tensor, probs:torch.Tensor) -> List[torch.Tensor]:
    """Computes the gradient of the log probability of the model output wrt to 
        the model parameters for a single label class
        
    Args:
        model (nn.Module): the model
        log_probs (torch.Tensor): the log probabilities of the model output (scalar)
        probs (torch.Tensor): the probabilities of the model output (scalar)
    Returns:
        List[torch.Tensor]: the gradient of the model output relative a single label class
    """
    grads = torch.autograd.grad(log_probs, model.parameters(), create_graph=True)
    grads = [grd.pow(2) * probs for grd in grads]
    
    return grads
