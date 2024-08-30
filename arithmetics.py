from torch import nn
from mergecraft.arithmetics.weights_wrapper import ModelTensorMapper
from transformers import pipeline

# Create pipelines for text classification using different models
gpt2 = pipeline('text-generation', model='openai-community/gpt2', device='cuda:0', framework='pt')
gpt2_toxic = pipeline('text-generation', model='heegyu/gpt2-toxic', device='cuda:0', framework='pt')

# Get the underlying models from the pipelines
model1 = gpt2.model
model2 = gpt2_toxic.model

# Create an instance of ArchitectureTensor to convert model weights to tensors
arch2tensor = ModelTensorMapper(model1)

# Convert model weights to tensors:
# The model's weights are mapped onto a simple torch tensor
# This operation allows the weights to be manipulated using arithmetic operations
weights1 = arch2tensor.to_tensor(model1)
weights2 = arch2tensor.to_tensor(model2)

print('Size of tensor:', weights1.size())

# Perform arithmetic operations on the weights
add = weights1 + weights2  # Addition
sub = weights1 - weights2  # Subtraction
mul = weights1 * weights2  # Multiplication
div = weights1 / weights2  # Division
mul_scalar = weights1 * 2  # Scalar multiplication
div_scalar = weights1 / 2  # Scalar division

# Convert the result of subtraction back to a model
task1 = arch2tensor.to_model(sub)


# Detoxfication of a toxic model
detoxified = arch2tensor.to_model(weights1 - 0.1 * (weights2 - weights1))

PROMPT = 'I\'m not saying you\'re stupid, but'
toxic_out  = gpt2_toxic(PROMPT, max_length=100, num_return_sequences=5)
normal_out = gpt2(PROMPT, max_length=100, num_return_sequences=5)

gpt2.model = detoxified
detoxified_out = gpt2(PROMPT, max_length=100, num_return_sequences=5)
print('Toxic model:\n', toxic_out)
print('Normal model:\n', normal_out)
print('Detoxified model:\n', detoxified_out)
