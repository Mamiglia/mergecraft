import torch
from torch import nn
from torch.utils.data import DataLoader
from src import fisher_matrix
from datasets import load_dataset
from transformers import pipeline, AutoModel
from tqdm import tqdm

# Load dataset and subset
SUBSET = 1 # None
dataset = load_dataset('glue', 'rte')
testset = dataset['validation'].select(range(SUBSET)) if SUBSET is not None else dataset['validation']
# Format dataset
inputs = [{"text": example['sentence1'], "text_pair": example['sentence2']} for example in testset]

fim = dict()
models = ['textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
for model in tqdm(models):
    # Load model
    pipe = pipeline('text-classification', model=model)

    # compute
    fim[model] = fisher_matrix(pipe, inputs)
    
    
    
for model, fim in fim.items():
    model = AutoModel.from_pretrained(model)
    
    
