# Lifted off pytorch tutorials as a useful guideline
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import torch
import math

dtype = torch.float
device = torch.device("cuda:0")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

#  Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + bx + cx**2 + dx**3
    y_pred = a + b*x + c*x**2 + d*x**3
    
    # Compute and print loss every 99 steps
    loss = (y_pred - y).pow(2).sum()
    if t%100==99:
        print(t, loss.item())

    # Backprop to compute gradients of a, b, c, d with respect to loss.
    loss.backward()

    with torch.no_grad():
	    # Update weights
	    a -= learning_rate * a.grad
	    b -= learning_rate * b.grad
	    c -= learning_rate * c.grad
	    d -= learning_rate * d.grad

	    a.grad = None
	    b.grad = None
	    c.grad = None
	    d.grad = None

print(f"Result: y = {a} + {b} x + {c} x^2 + {d} x^3")