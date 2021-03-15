# # Lifted off pytorch tutorials as a useful guideline
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

# Legendre Polynomial, degree 3
# P_3(x) = (1/2)(5x^3 - 3x)

import torch
import math

class LegendrePolynomial3(torch.autograd.Function):
    """
    We can implement custom autograd functions by subclassing 
    torch.autograd.Function and implementing the forward and backward 
    passes which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input_tensor):
        """
        In the forward pass we receive a Tensor containing the input_tensor and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input_tensor)
        return 0.5 * (5 * input_tensor**3 - 3 * input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss with
        respect to the input.
        """
        input_tensor, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input_tensor**2 - 1)

dtype = torch.float
device = torch.device("cuda:0")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for t in range(2000):
    # to apply the function, we have to use the Function.apply method.
    P3 = LegendrePolynomial3.apply

    y_pred = a + b * P3(c + d * x)

    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')