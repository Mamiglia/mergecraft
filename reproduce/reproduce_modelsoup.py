import time
from src import WeightedMerger
from datasets import load_dataset
from transformers import pipeline

# Load dataset and subset
SUBSET = None
DATASET = 'rte' # 'sst2', ''
SPLIT = 'validation'

if DATASET == 'rte':
    models = ['textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
else:
    models = ['aviator-neural/bert-base-uncased-sst2','howey/bert-base-uncased-sst2', 'yoshitomo-matsubara/bert-base-uncased-sst2', 'ikevin98/bert-base-uncased-finetuned-sst2', 'TehranNLP-org/bert-base-uncased-cls-sst2']

pipe = pipeline('text-classification', model=models[0], device='cpu', framework='pt')

t0 = time.time()
merger = WeightedMerger(models, task='text-classification')
pipe.model = merger.merge()
print('Merging completed. Time elapsed:', time.time()-t0)
# Save the merged weights
pipe.model.save_pretrained(f'./artifacts/merged_weights_soup_{DATASET}_{SPLIT}.pt')

print('Evaluating the merged model')
from src import evaluate_glue_pipeline
res = evaluate_glue_pipeline(pipe, 'rte')
print(res)