import torch
from evaluate import load 
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyPairDataset, KeyDataset

def evaluate_glue_pipeline(pipeline, dataset_name):
    dataset = load_dataset('glue', dataset_name)
    metric = load('glue', dataset_name)

    # Prepare the dataset
    if dataset_name == 'rte':
        valset = KeyPairDataset(dataset['validation'], 'sentence1', 'sentence2')
    else:
        valset = KeyDataset(dataset['validation'], 'sentence')
    
    # Evaluate: compute the accuracy of the model on the RTE dataset
    predicted = pipeline(valset)
    predicted = [0 if p['label'] == 'LABEL_0' else 1 for p in predicted]
    metric.add_batch(predictions=predicted, references=dataset['validation']['label'])
    return metric.compute()

if __name__=='__main__':
    PATH = './artifacts/merged_weights_rte'
    pipe = pipeline('text-classification', 
                    model=PATH, 
                    tokenizer='textattack/bert-base-uncased-RTE', 
                    device='cuda:0', framework='pt')

    res = evaluate_glue_pipeline(pipe, 'rte')
    print(res)
