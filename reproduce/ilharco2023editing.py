import time
from src import TaskMerger
from transformers import pipeline

# Load dataset and subset
SUBSET = None
DATASET = 'rte' # 'sst2', ''
SPLIT = 'validation'

if DATASET == 'rte':
    models = ['google-bert/bert-base-uncased', 'textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
else:
    models = ['google-bert/bert-base-uncased', 'aviator-neural/bert-base-uncased-sst2','howey/bert-base-uncased-sst2', 'yoshitomo-matsubara/bert-base-uncased-sst2', 'ikevin98/bert-base-uncased-finetuned-sst2', 'TehranNLP-org/bert-base-uncased-cls-sst2']

pipe = pipeline('text-classification', model=models[0], device='cpu', framework='pt')

t0 = time.time()
merger = TaskMerger(models, task='text-classification', base_index=0, passthrough_layers=['classifier.bias', 'classifier.weight'])
pipe.model = merger.merge(signs=[1, 1, 1, 1, 1])
print('Merging completed. Time elapsed:', time.time()-t0)
# Save the merged weights
pipe.model.save_pretrained(f'./artifacts/merged_weights_task_{DATASET}_{SPLIT}.pt')

print('Evaluating the merged model')
from src import evaluate_glue_pipeline
res = evaluate_glue_pipeline(pipe, 'rte')
print(res)
