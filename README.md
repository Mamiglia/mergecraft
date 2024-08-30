# Merging through arithmetics

A simple library for model merging. This library aim is to implement various merging functions through multiple mechanisms, and make it easy to implement other merging methods yourself. 

To begin with you can do
```python
models = ['textattack/bert-base-uncased-RTE', 'yoshitomo-matsubara/bert-base-uncased-rte', 'Ruizhou/bert-base-uncased-finetuned-rte', 'howey/bert-base-uncased-rte', 'anirudh21/bert-base-uncased-finetuned-rte']

merger = WeightedMerger(models, task='text-classification')
merged_model = merger.merge()
pipe = pipeline('text-classification', model=merged_model)

```