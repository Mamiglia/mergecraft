from typing import Iterable, List, Optional

from torch import Tensor

from mergecraft.merging.base import layer_merge

@layer_merge
def task(models: Iterable[Tensor], base_index:int=0, signs: Optional[List[int]]=None, **_) -> Tensor:
    '''Implements the model merging algorithm from the paper: 
        "Editing Models with Task Arithmetic", Ilharco et al. 2023
    
    Args:
    - models (Iterable): list of models to be merged, fine-tuned and baseline
    - base_index (int): index of the baseline model which was the base for fine-tuning
    - signs (Optional[List[int]]): list of signs for each model, indicating the direction 
        of the task vector, i.e. whether to add or subtract the task vector from the baseline model

    Returns:
    - Merged model
    '''
    pretrained = models.pop(base_index)
        
    if signs is None:
        signs = [1] * len(models)
    assert len(signs) == len(models), 'Number of signs should be equal to the number of models'
    assert all(s in [-1, 1] for s in signs), 'Sign should be either -1 or 1'
        
    tasks = [model - pretrained for model in models]  # list of task vectors
    tasks = [task * s for task, s in zip(tasks, signs)] 
    merged_task = sum(tasks) / len(tasks)
    return pretrained + merged_task
