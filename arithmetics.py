import torch
from torch import nn
from src.weights_wrapper import Weights
from datasets import load_dataset
from transformers import pipeline


# Load model
models = ['textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
pipe1 = pipeline('text-classification', model=models[0], device='cuda:0', framework='pt')
pipe2 = pipeline('text-classification', model=models[1], device='cuda:0', framework='pt')

model1 = pipe1.model
model2 = pipe2.model

weights1 = Weights(model1.named_parameters())
weights2 = Weights(model2.named_parameters())

add = weights1 + weights2
sub = weights1 - weights2   
mul = weights1 * weights2
div = weights1 / weights2
mul_scalar = weights1 * 2
div_scalar = weights1 / 2

