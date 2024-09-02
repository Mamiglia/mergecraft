import time
from mc import *
from datasets import load_dataset
from transformers import pipeline

# Load dataset and subset
SUBSET = None
DATASET = 'mrpc' # 'sst2', ''
SPLIT = 'validation'

match DATASET:
    case 'rte':
        MODELS = ['google-bert/bert-base-uncased', 'textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
    case 'sst2':
        MODELS = ['google-bert/bert-base-uncased', 'aviator-neural/bert-base-uncased-sst2','howey/bert-base-uncased-sst2', 'yoshitomo-matsubara/bert-base-uncased-sst2', 'doyoungkim/bert-base-uncased-finetuned-sst2', 'TehranNLP-org/bert-base-uncased-cls-sst2']
    case 'mrpc':
        MODELS = ['google-bert/bert-base-uncased', 'textattack/bert-base-uncased-MRPC', 'yoshitomo-matsubara/bert-base-uncased-mrpc', 'Serjssv/bert-base-uncased-mrpc', 'Ruizhou/bert-base-uncased-finetuned-mrpc', 'TehranNLP-org/bert-base-uncased-mrpc-2e-5-42']
    case _:
        raise ValueError(f'Dataset {DATASET} not supported')
    
for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    t0 = time.time()
    merged_pipe = slerp(MODELS[1:3], task='text-classification', tau=t)
    dt = time.time()-t0
    print('Merging completed. Time elapsed:', dt)
    # Save the merged weights
    merged_pipe.model.save_pretrained(f'./artifacts/merged_weights_slerp_{DATASET}_{SPLIT}')

    merged_pipe = pipeline('text-classification',
                        model=f'./artifacts/merged_weights_slerp_{DATASET}_{SPLIT}', 
                        tokenizer=MODELS[0],
                        device='cuda:0', framework='pt')

    print('Evaluating the merged model')
    from mergecraft import evaluate_glue_pipeline
    res = evaluate_glue_pipeline(merged_pipe, DATASET)
    print('T:', t, res)

