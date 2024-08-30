from mc import fisher
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyPairDataset, KeyDataset
from datasets import load_dataset
import time

# Load dataset and subset
SUBSET = None
DATASET = 'rte' # 'sst2', ''
SPLIT = 'validation'

dataset = load_dataset('glue', DATASET)
print(dataset)
subset = dataset[SPLIT].select(range(SUBSET)) if SUBSET else dataset[SPLIT]
if DATASET == 'rte':
    subset = KeyPairDataset(subset, 'sentence1', 'sentence2')
    models = ['textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
else:
    subset = KeyDataset(subset, 'sentence')
    models = ['aviator-neural/bert-base-uncased-sst2','howey/bert-base-uncased-sst2', 'yoshitomo-matsubara/bert-base-uncased-sst2', 'ikevin98/bert-base-uncased-finetuned-sst2', 'TehranNLP-org/bert-base-uncased-cls-sst2']

pipe = pipeline('text-classification', model=models[0], device='cpu', framework='pt')

t0 = time.time()
merged_pipe = fisher(
    models, 
    task='text-classification',
    pipelines=[pipeline('text-classification', model=model, device='cuda:0') for model in models],
    dataset=subset
    )
dt = time.time()-t0
print('Merging completed. Time elapsed:', dt)
# Save the merged weights
merged_pipe.model.save_pretrained(f'./artifacts/merged_weights_fisher_{DATASET}_{SPLIT}')

merged_pipe = pipeline('text-classification', 
                    model=f'./artifacts/merged_weights_fisher_{DATASET}_{SPLIT}', 
                    tokenizer=models[0], 
                    device='cuda:0', framework='pt')

print('Evaluating the merged model')
from mergecraft import evaluate_glue_pipeline
res = evaluate_glue_pipeline(merged_pipe, DATASET)
print(res)

import json
record = {
    'method': 'fisher',
    'dataset': DATASET,
    'split': SPLIT,
    'directory': f'./artifacts/merged_weights_fisher_{DATASET}_{SPLIT}',
    'time': dt,
    **res
}
with open(f'./artifacts/merged_weights_fisher_{DATASET}_{SPLIT}/record.json', "w") as fp:
    json.dump(record , fp) 

