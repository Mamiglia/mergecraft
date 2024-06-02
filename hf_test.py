import torch
from torch import nn
from torch.utils.data import DataLoader
from src.fisher import fisher_matrix

# import a simple pretrained model from huggingface
# and its dataset
# finally apply fisher_matrix to the model and dataset


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset

# Load the DistilBert tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Load the IMDB dataset
dataset = load_dataset('imdb')

def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Apply the function to the text data in the dataset
dataset = dataset.map(encode, batched=True)

fisher_matrix(model, DataLoader(dataset['train'], batch_size=1), criterion=nn.CrossEntropyLoss())