#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: activation.py

import numpy as np


def sigmoid(z):
    """The sigmoid function."""
    sigmoid_result = 1.0/(1+np.exp(-z))
    return sigmoid_result
    # return 0

def sigmoid_prime(z):
    # """Derivative of the sigmoid function."""
	return z*(1-z)
    # return 0
