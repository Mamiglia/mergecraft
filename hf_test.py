import torch
from torch import nn
from torch.utils.data import DataLoader
from src import fisher_matrix
from datasets import load_dataset
from transformers import pipeline

# Load model
models = ['textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
pipe = pipeline('text-classification', model=models[0], device='cuda:0', framework='pt')

# Load dataset and subset
SUBSET = None # 10
dataset = load_dataset('glue', 'rte')
testset = dataset['validation'].select(range(SUBSET)) if SUBSET else dataset['validation']
# Format dataset
inputs = [{"text": example['sentence1'], "text_pair": example['sentence2']} for example in testset]

# compute
print('begin computation')
fim = fisher_matrix(pipe, inputs)
print('end computation')
# print(fim)