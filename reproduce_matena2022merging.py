import torch
from torch import nn
from torch.utils.data import DataLoader
from src import fisher_matrix
from datasets import load_dataset
from transformers import pipeline, AutoModel
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyPairDataset, KeyDataset
from src.weights_wrapper import Weights

# Load dataset and subset
SUBSET = None
dataset = load_dataset('glue', 'sst2')
print(dataset)
testset = dataset['validation'].select(range(SUBSET)) if SUBSET else dataset['validation']
# testset = KeyPairDataset(testset, 'sentence1', 'sentence2')
testset = KeyDataset(testset, 'sentence')
# Format dataset
fim = dict()
models = ['aviator-neural/bert-base-uncased-sst2','howey/bert-base-uncased-sst2', 'yoshitomo-matsubara/bert-base-uncased-sst2', 'ikevin98/bert-base-uncased-finetuned-sst2', 'TehranNLP-org/bert-base-uncased-cls-sst2']
# models = ['textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']
# initialize weights
merged_weights = Weights.from_model( pipeline('text-classification', model=models[0], framework='pt').model, zero_init=True)
merged_weights.to('cpu')

for model in tqdm(models):
    # Load model
    pipe = pipeline('text-classification', model=model, device='cuda:0', framework='pt')

    # compute
    fim[model] = fisher_matrix(pipe, testset).to('cpu')
    merged_weights += fim[model] * Weights.from_model(pipe.model.to('cpu'))

total_fim = sum(fim.values(), start=Weights.from_model(pipe.model.to('cpu'), zero_init=True))
merged_weights /= total_fim


# Save the merged weights
from evaluate import evaluator, load
final_weights = merged_weights.named_parameters
for name, param in pipe.model.named_parameters():
    if not param.requires_grad:
        continue
    if torch.any(torch.isnan(final_weights[name])):
        print(f'NaNs in {name}')
    param.data = final_weights[name].to('cuda:0')
pipe.model.to('cuda:0')
    
pipe.model.save_pretrained('./artifacts/merged_weights_sst2')
# pipe = pipeline('text-classification', model='./artifacts/merged_weights.pt', tokenizer=models[0], device='cuda:0')
metric = load('glue', 'sst2')
# valset = KeyPairDataset(dataset['validation'], 'sentence1', 'sentence2')
valset = KeyDataset(dataset['validation'], 'sentence')
# Evaluate: compute the accuracy of the model on the RTE dataset
predicted = pipe(valset)
predicted = [0 if p['label'] == 'LABEL_0' else 1 for p in predicted]
metric.add_batch(predictions=predicted, references=dataset['validation']['label'])
accuracy = metric.compute()['accuracy']
print(f'Accuracy: {accuracy}')

