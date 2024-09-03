# Mergecraft

**Mergecraft** is a Python library designed to simplify the process of merging machine learning models. Whether you're combining weights from different models or experimenting with novel merging strategies, Mergecraft provides the tools and decorators to make this process seamless and efficient.

## Features

- **@model_merge**: A decorator that handles the complex operations of model loading, casting, and skipped layers, allowing you to focus on the core merging logic.
- **@dict_map**: Simplifies the process of applying layer-wise merging functions across an entire model.
- **@layer_merge**: A powerful decorator that combines `@dict_map` and `@model_merge`, making it easier to implement straightforward merging methods.

## Quick Start

### Running an Evaluation

You can evaluate your merged model on tasks like the RTE task from the GLUE benchmark:

```python
models = [
    'textattack/bert-base-uncased-RTE',
    'yoshitomo-matsubara/bert-base-uncased-rte',
    'Ruizhou/bert-base-uncased-finetuned-rte',
    'howey/bert-base-uncased-rte',
    'anirudh21/bert-base-uncased-finetuned-rte'
]

from mergecraft import dare
dare(models, task='text-classification')
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
