import torch
from evaluate import load 
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyPairDataset, KeyDataset

def evaluate_glue_pipeline(pipeline, dataset_name, split='validation'):
    dataset = load_dataset('glue', dataset_name)
    metric = load('glue', dataset_name)

    # Prepare the dataset
    if dataset_name in ['rte', 'mrpc']:
        splitset = KeyPairDataset(dataset[split], 'sentence1', 'sentence2')
    elif dataset_name in ['sst2']:
        splitset = KeyDataset(dataset[split], 'sentence')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    # Evaluate: compute the accuracy of the model on the dataset
    predicted = pipeline(splitset)
    predicted = [0 if p['label'] == 'LABEL_0' else 1 for p in predicted]
    return metric.compute(predictions=predicted, references=dataset[split]['label'])

if __name__=='__main__':
    PATH = './artifacts/merged_weights_rte'
    pipe = pipeline('text-classification', 
                    model=PATH, 
                    tokenizer='textattack/bert-base-uncased-RTE', 
                    device='cuda:0', framework='pt')

    res = evaluate_glue_pipeline(pipe, 'rte')
    print(res)
