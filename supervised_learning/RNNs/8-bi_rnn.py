#!/usr/bin/env python3
"""
Defines function that performs forward propagation for bidirectional RNN
"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN

    Parameters:
    - bi_cell: An instance of BidirectionalCell
    - X: numpy.ndarray of shape (t, m, i), data to be used
        - t: maximum number of time steps
        - m: batch size
        - i: dimensionality of data
    - h_0: numpy.ndarray of shape (m, h), initial hidden state (forward direction)
    - h_t: numpy.ndarray of shape (m, h), initial hidden state (backward direction)

    Returns:
    - H: numpy.ndarray of shape (t, m, 2*h), concatenated hidden states
    - Y: numpy.ndarray of shape (t, m, o), outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    # Initialize hidden states
    h_forward = np.zeros((t, m, h))
    h_backward = np.zeros((t, m, h))

    # Forward propagation (left-to-right)
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        h_forward[step] = h_prev

    # Backward propagation (right-to-left)
    h_prev = h_t
    for step in range(t - 1, -1, -1):
        h_prev = bi_cell.backward(h_prev, X[step])
        h_backward[step] = h_prev

    # Concatenate hidden states
    H = np.concatenate((h_forward, h_backward), axis=2)

    # Compute outputs for each concatenated hidden state
    Y = np.zeros((t, m, bi_cell.Wy.shape[1]))
    for step in range(t):
        Y[step] = bi_cell.output(H[step])

    return H, Y
