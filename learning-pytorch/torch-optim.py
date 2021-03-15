# We've been updating our weights manually, but it might be wiser to use optim,
# especially when we begin to deal with more weights.

import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1,2,3])
xx = x.unsqueeze(-1).pow(p)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(2000):
	y_pred = model(xx)

	loss = loss_fn(y_pred, y)
	if t % 100 == 99:
		print(t, loss.item())

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

linear_layer = model[0]
print(
	f'Result: y = {linear_layer.bias.item()} \
+ {linear_layer.weight[:, 0].item()} x \
+ {linear_layer.weight[:, 1].item()} x^2 \
+ {linear_layer.weight[:, 2].item()} x^3')