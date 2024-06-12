import torch
from torch import nn
from src.weights_wrapper import ArchitectureTensor
from datasets import load_dataset
from transformers import pipeline


# Load model
models = ['textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
pipe1 = pipeline('text-classification', model=models[0], device='cuda:0', framework='pt')
pipe2 = pipeline('text-classification', model=models[1], device='cuda:0', framework='pt')

model1 = pipe1.model
model2 = pipe2.model

arch2tensor = ArchitectureTensor(model1)
weights1 = arch2tensor.to_tensor(model1)
weights2 = arch2tensor.to_tensor(model2)

add = weights1 + weights2
sub = weights1 - weights2   
mul = weights1 * weights2
div = weights1 / weights2
mul_scalar = weights1 * 2
div_scalar = weights1 / 2

task1 = arch2tensor.to_model(sub)
