# Deep Neural Network Engine — Image Classification

Custom neural network built at the linear algebra level, applied to binary image classification. Implements the full computational graph of a deep network without relying on any ML framework.

## Problem

High-level deep learning frameworks abstract away the mechanics that determine why networks converge — or don't. Building from matrix operations upward makes architectural decisions principled and debugging tractable.

## Implementation

Configurable L-layer architecture supporting arbitrary depth:

- **He initialization** — prevents vanishing/exploding gradients in deep networks
- **Vectorized forward propagation** — full mini-batch processed as matrix operations across L layers
- **Activation functions** — ReLU in hidden layers, sigmoid at output; analytic derivatives for backprop
- **Backward propagation** — chain rule applied layer by layer using cached activations from forward pass
- **Gradient descent** — parameter updates with configurable learning rate and iteration count

## Results

Applied to binary image classification, achieving >80% accuracy on held-out data with a 5-layer network, compared to ~65% for a logistic regression baseline.

## Stack

`Python` `NumPy` `matplotlib` `h5py`
