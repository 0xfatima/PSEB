"""
model.py - the digit reader network, shared by train.py and api.py so that
training and serving always build exactly the same architecture.
"""

from torch import nn


def build_model(sizes):
    """Build an MLP from a list of layer sizes, e.g. [64, 64, 10].

    64 inputs (8x8 pixels) -> hidden layer(s) with ReLU -> 10 digit scores.
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:          # no ReLU after the output layer
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
