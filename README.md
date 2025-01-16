# forch

To better understand PyTorch, I implemented a simple clone of it.
Instead of `torch`, just import `forch`.
There is no Autograd, so an explicit backward function is required.

A notebook using forch for training can be found [here](main.ipynb).

## Functionality

Layers:
- Linear

Loss Functions:
- Mean Squared Error

Activation Functions:
- Sigmoid
- ReLU

Optimizers:
- SGD