#!/usr/bin/env python3
import numpy as np

class BidirectionalCell:
    def __init__(self, i, h, o):
        self.Whf = np.random.randn(i + h, h)  # Weights for forward direction
        self.Whb = np.random.randn(i + h, h)  # Weights for backward direction
        self.Wy = np.random.randn(2 * h, o)   # Weights for output
        self.bhf = np.zeros((1, h))           # Bias for forward direction
        self.bhb = np.zeros((1, h))           # Bias for backward direction
        self.by = np.zeros((1, o))            # Bias for output

    def forward(self, h_prev, x_t):
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        y = np.dot(H, self.Wy) + self.by
        return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
