# Mergecraft

**Mergecraft** is a Python library designed to simplify the process of merging machine learning models. Whether you're combining weights from different models or experimenting with novel merging strategies, Mergecraft provides the tools and decorators to make this process seamless and efficient.

## Features

- Easy-to-use interface for model merging
- Support for various merging paradigms (TIES, SLERP, DARE, etc.)
- Seamless integration with Hugging Face models
- Extensible architecture for implementing custom merging methods
- Efficient handling of model loading and conversion

## Quick Start

### Running an Evaluation

You can evaluate your merged model on tasks like the RTE task from the GLUE benchmark:

```python
models = [
    'google-bert/bert-base-uncased',
    'textattack/bert-base-uncased-RTE',
    'yoshitomo-matsubara/bert-base-uncased-rte',
    'Ruizhou/bert-base-uncased-finetuned-rte',
    'howey/bert-base-uncased-rte',
    'anirudh21/bert-base-uncased-finetuned-rte'
]

# List of layers where DARE should be skipped
# because these layers are randomly initialized during finetuning
rnd_layers = ['classifier.weight', 'classifier.bias']

merged = mergecraft.dare(models, passthrough=rnd_layers, base_index=0)
```

### Example: Median Merger

Below is an example of how to implement a median-based model merging strategy using Mergecraft:

```python
import torch
from mergecraft import layer_merge

@layer_merge
def median_merge(models: List[torch.Tensor], **kwargs) -> torch.Tensor:
    return torch.median(torch.stack(models), dim=0).values
```


## Contributing

Contributions are welcome! If you’d like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcomed.
