import torch
from torch import nn
from torch.utils.data import DataLoader
from src import fisher_matrix
from datasets import load_dataset
from transformers import pipeline
from time import time
from transformers.pipelines.pt_utils import KeyPairDataset

# Load model
models = ['textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
pipe = pipeline('text-classification', model=models[0], device='cuda:0', framework='pt')

# Load dataset and subset
SUBSET = None # 10
dataset = load_dataset('glue', 'rte')
testset = dataset['validation'].select(range(SUBSET)) if SUBSET else dataset['validation']
# Format dataset
testset = KeyPairDataset(testset, 'sentence1', 'sentence2') 
print(testset[0])
print(pipe(testset[0]))
# compute
t0 = time()
print('begin computation.')
fim = fisher_matrix(pipe, testset)
print('end computation. This took:', time()-t0)

print('Is FIM on cuda?', fim.weights[0].is_cuda)