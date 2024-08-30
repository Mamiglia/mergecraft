import argparse
import json
import os
import time

import pandas as pd
from datasets import load_dataset
from lib import *
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset, KeyPairDataset

from mc import *

# Load dataset and subset
SUBSET = None
DATASET = 'mrpc'#'rte' # 'sst2', 'mrpc'
SPLIT = 'validation' # 'validation', 'test'

dataset = load_dataset('glue', DATASET)
print(dataset)
subset = dataset[SPLIT].select(range(SUBSET)) if SUBSET else dataset[SPLIT]
match DATASET:
    case 'rte':
        subset = KeyPairDataset(subset, 'sentence1', 'sentence2')
        MODELS = ['google-bert/bert-base-uncased', 'textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
    case 'sst2':
        subset = KeyDataset(subset, 'sentence')
        MODELS = ['google-bert/bert-base-uncased', 'aviator-neural/bert-base-uncased-sst2','howey/bert-base-uncased-sst2', 'yoshitomo-matsubara/bert-base-uncased-sst2', 'doyoungkim/bert-base-uncased-finetuned-sst2', 'TehranNLP-org/bert-base-uncased-cls-sst2']
    case 'mrpc':
        subset = KeyPairDataset(subset, 'sentence1', 'sentence2')
        MODELS = ['google-bert/bert-base-uncased', 'textattack/bert-base-uncased-MRPC', 'yoshitomo-matsubara/bert-base-uncased-mrpc', 'Serjssv/bert-base-uncased-mrpc', 'Ruizhou/bert-base-uncased-finetuned-mrpc', 'TehranNLP-org/bert-base-uncased-mrpc-2e-5-42']
    case _:
        raise ValueError(f'Dataset {DATASET} not supported')
    
METHODS = {
    'task': task,
    'soup': soup,
    'fisher': fisher,
    'dare': dare,
    'stock': stock,
    'ties': ties
}
# METHODS.update({m:m for i,m in enumerate(MODELS)})

KWARGS = {}
KWARGS.update({
    'task': {'base_index': 0, 'passthrough_layers': ['classifier.bias', 'classifier.weight']},
    'soup': {},
    'fisher': dict(
        pipelines=[pipeline('text-classification', model=model, device='cuda:0') for model in MOEDLS],
        dataset=subset),
    'dare': dict(p=0.4, base_index=0, passthrough_layers=['classifier.bias', 'classifier.weight']),
    'stock': {'base_index': 0, 'passthrough_layers': ['classifier.bias', 'classifier.weight']},
    'ties': dict(k=0.5, base_index=0, passthrough_layers=['classifier.bias', 'classifier.weight']),
})
RECORDS = []


pipe = pipeline('text-classification', model=MODELS[-1], device='cpu', framework='pt')

for name, merge_method in METHODS.items():
    print(f'Running method {name}')
    t0 = time.time()
    if '/' not in name:
        models = MODELS if 'base_index' in KWARGS[name].keys() else MODELS[1:]
        pipe = merge_method(models, task='text-classification', **KWARGS[name])
        # Save the merged weights
        pipe.model.save_pretrained(f'./artifacts/merged_weights_{name}_{DATASET}_{SPLIT}')
        pipe = pipeline('text-classification', model=f'./artifacts/merged_weights_{name}_{DATASET}_{SPLIT}', device='cuda:1', framework='pt')
    else:
        pipe = pipeline('text-classification', model=name, device='cuda:1', framework='pt')
    dt = time.time()-t0
    print('Merging completed. Time elapsed:', dt)

    print('Evaluating the merged model')
    res = evaluate_glue_pipeline(pipe, DATASET, split=SPLIT)
    print(res)

    record = {
        'method': name,
        'dataset': DATASET,
        'split': SPLIT,
        'directory': f'./artifacts/merged_weights_{name}_{DATASET}_{SPLIT}',
        'time': dt, 
        **res
    }
    RECORDS.append(record)
 
    # RECORDS is a list of dictionaries, each containing the method, dataset, split, directory, accuracy, and time of the merge
    # We can convert this to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(RECORDS)
    print(df)
    # Save the DataFrame to a CSV file for later analysis with time
    df.to_csv(f'./artifacts/merged_weights_{DATASET}_{SPLIT}_{time.time()//3600}.csv', index=False)

