# Lifted off pytorch tutorials as a useful guideline
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

# When building neural networks, we frequently arrange computation into *layers*,
# some of which have learnable paramaters which will be optimized during learning.

# In PyTorch, nn package serves this same purpose. nn package defines a set of Modules,
# which are roughly equivalent to neural network layers.

# A Module receives input Tensors and computes output Tensors, but 
# may also hold internal state such as Tensors containing learnable parameters.
# nn package defines set of useful loss functions that are commonly used when training neural networks.

import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
# unsqueeze it to "batchify" it, then get three diff values with powers in p
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
	torch.nn.Linear(3, 1),
	torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):
	y_pred = model(xx)

	loss = loss_fn(y_pred, y)
	if t % 100 == 99:
		print(t, loss.item())

	model.zero_grad()
	loss.backward()

	with torch.no_grad():
		for param in model.parameters():
			param -= learning_rate * param.grad

# You can access the first layer of sequential by indexing.
linear_layer = model[0]

# For linear layer, its parameters are stored as `weight` and `bias`.
print(
	f'Result: y = {linear_layer.bias.item()} \
+ {linear_layer.weight[:, 0].item()} x \
+ {linear_layer.weight[:, 1].item()} x^2 \
+ {linear_layer.weight[:, 2].item()} x^3')