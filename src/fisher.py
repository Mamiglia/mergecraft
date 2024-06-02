from typing import Dict, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from einops import reduce, einsum
from tqdm import tqdm

def fisher_matrix(model : nn.Module, dataloader : DataLoader, criterion: nn.Module, device='cpu') -> Dict[str, torch.Tensor]:
    '''compute Fisher Information Matrix diagonal approximation'''
    """
            Compute the Diagonal of the Hessian Matrix for the given Dataloader. 
            Note: In the implementation of the Diagonal of the Hessian Matrix we have weighted the second order derivatives by the 
                  confidence of the samples wrt to the class.The significance of multiplying the squared gradient by the probability for 
                  that class lies in estimating the curvature of the loss function with respect to the parameters. The Hessian matrix 
                  represents the second-order derivatives of the loss function with respect to the parameters and provides information 
                  about the local geometric properties of the loss surface.

                  In the context of training a model for classification tasks, the probability for a particular class in the mini-batch 
                  measures the likelihood of that class being the correct label. By multiplying the squared gradient by this probability
                  , the algorithm assigns a higher weight to gradients associated with classes that are more likely to be correct. This 
                  weighting reflects the importance of each class in determining the curvature of the loss function.

                  Remember while doing batch gradient we approximate the gradient.
    """
    model.eval()
    model.to(device)
    assert dataloader.batch_size == 1, "Batch size should be 1 for computing Hessian Matrix"

    # hessian for each sample
    hessians = [hessian_batch(model, batch, criterion, device) for batch in dataloader]
    
    num_tensors = len(hessians[0])
    hessian = [sum([h[i] for h in hessians]) for i in range(tqdm(num_tensors))]
    
    return hessian

def hessian_batch(model :nn.Module, batch: Tuple[torch.Tensor], criterion:nn.Module, device) -> Dict[str, torch.Tensor]:
    '''compute Hessian Matrix diagonal approximation for a batch'''
    *input, label = [item.to(device) for item in batch] # 1x...
    output = model(**input) # ...xC
    probs = F.softmax(output, dim=-1) # ...xC
    loss = criterion(output, label)

    return hess(model, loss, probs)

def hess(model: nn.Module, loss: torch.Tensor, probs: torch.Tensor) -> Dict[str, torch.Tensor]:
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    grads = [compute_dx2d(g, probs) for g in grads]
    return grads

@torch.no_grad()
def compute_dx2d(grd:torch.Tensor, probs:torch.Tensor):
    shape = grd.shape
    grd = grd.detach()
    grd = torch.pow(grd,2).flatten()
    # compute expected value over output classes
    grd = einsum(grd, probs, 'h, ... c -> h c').sum(dim=-1)
    
    return grd.reshape(shape)
    
