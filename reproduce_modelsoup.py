import torch
from torch import nn
from torch.utils.data import DataLoader
from src import fisher_matrix, ArchitectureTensor, weighted_merging
from datasets import load_dataset
from transformers import pipeline, AutoModel
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyPairDataset, KeyDataset

# Load dataset and subset
SUBSET = None
DATASET = 'rte'
SPLIT = 'validation'

dataset = load_dataset('glue', DATASET)
print(dataset)
testset = dataset[SPLIT].select(range(SUBSET)) if SUBSET else dataset[SPLIT]
if DATASET == 'rte':
    testset = KeyPairDataset(testset, 'sentence1', 'sentence2')
    models = ['textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
else:
    testset = KeyDataset(testset, 'sentence')
    models = ['aviator-neural/bert-base-uncased-sst2','howey/bert-base-uncased-sst2', 'yoshitomo-matsubara/bert-base-uncased-sst2', 'ikevin98/bert-base-uncased-finetuned-sst2', 'TehranNLP-org/bert-base-uncased-cls-sst2']

modules = dict()

for model in tqdm(models):
    # Load model
    pipe = pipeline('text-classification', model=model, device='cpu', framework='pt')
    modules[model] = pipe.model


fisher_merged = weighted_merging(modules.values(), [1 for _ in modules])
pipe.model = fisher_merged

# Save the merged weights
pipe.model.save_pretrained('./artifacts/merged_weights_soup_rte')


from src import evaluate_glue_pipeline
res = evaluate_glue_pipeline(pipe, 'rte')
print(res)
