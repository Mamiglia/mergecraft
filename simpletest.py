import torch
from torch import nn
from torch.utils.data import DataLoader 
from src.fisher import fisher_matrix
import random


model = nn.Sequential(
    nn.Linear(10, 11),
    nn.ReLU(),
    nn.Linear(11, 3)
)
dataset = [(torch.randn(10), random.randint(0,2)) for _ in range(100)]
loader = DataLoader(dataset, batch_size=1)

L = fisher_matrix(model, loader, criterion=nn.CrossEntropyLoss())
print(L)