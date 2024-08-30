import time
from lib import stock
from datasets import load_dataset
from transformers import pipeline
import os

# Load dataset and subset
SUBSET = None
DATASET = 'rte' # 'sst2', ''
SPLIT = 'validation'

if DATASET == 'rte':
    models = ['google-bert/bert-base-uncased', 'textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
else:
    models = ['google-bert/bert-base-uncased', 'aviator-neural/bert-base-uncased-sst2','howey/bert-base-uncased-sst2', 'yoshitomo-matsubara/bert-base-uncased-sst2', 'ikevin98/bert-base-uncased-finetuned-sst2', 'TehranNLP-org/bert-base-uncased-cls-sst2']

pipe = pipeline('text-classification', model=models[0], device='cpu', framework='pt', passthrough_layers=['classifier.weight', 'classifier.bias'])

t0 = time.time()
merged_pipe = stock(models, task='text-classification', 
                    base_index=0, passthrough_layers=['classifier.weight', 'classifier.bias'])
dt = time.time()-t0
print('Merging completed. Time elapsed:', dt)
# Save the merged weights
merged_pipe.model.save_pretrained(f'./artifacts/merged_weights_dare_{DATASET}_{SPLIT}')

merged_pipe = pipeline('text-classification', 
                    model=f'./artifacts/merged_weights_dare_{DATASET}_{SPLIT}', 
                    tokenizer=models[0], 
                    device='cuda:0', framework='pt')

print('Evaluating the merged model')
from src import evaluate_glue_pipeline
res = evaluate_glue_pipeline(merged_pipe, DATASET)
print(res)

import json
record = {
    'method': 'DARE',
    'dataset': DATASET,
    'split': SPLIT,
    'directory': f'./artifacts/merged_weights_dare_{DATASET}_{SPLIT}',
    'time': dt,
    **res
}
with open(f'./artifacts/merged_weights_dare_{DATASET}_{SPLIT}/record.json', "w") as fp:
    json.dump(record , fp) 
