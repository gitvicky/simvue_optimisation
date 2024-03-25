#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23rd March, 2024

Utilities for performing surrogate based optimisaiton. 
"""
import numpy as np
import torch 
import torch.nn as nn 
import gpytorch 

def Tensorize(*args):
    """
    Stack an arbitrary number of variables along a new axis.
    
    Args:
        *args: Arbitrary number of variables to stack.
        
    Returns:
        torch.Tensor: Torch tensor of the stacked input variables 
    """
    # Convert input arguments to numpy arrays
    arrays = [np.array(arg) for arg in args]
    
    # Check if all arrays have the same shape
    shapes = [arr.shape for arr in arrays]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All input arrays must have the same shape.")
    
    # Stack arrays along a new axis
    stacked_array = np.stack(arrays, axis=-1)
    stacked_tensor = torch.tensor(stacked_array, dtype=torch.float32)
    return stacked_tensor

#normalization, rangewise but across the full domain 
class Normalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(Normalizer, self).__init__()
        mymin = torch.min(x, axis=0)[0]
        mymax = torch.max(x, axis=0)[0]

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        x = self.a*x + self.b
        return x

    def decode(self, x):
        x = (x - self.b)/self.a
        return x

# Unit Gaussian Standardisation
class Standardizer(object):
    def __init__(self, x, eps=0.01):
        super(Standardizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

# Fully Connected Network or a Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.tanh):
        super(MLP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.act_func = activation

        self.layers = nn.ModuleList()

        self.layer_input = nn.Linear(self.in_features, self.num_neurons)

        for ii in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.num_neurons, self.num_neurons))
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

    def forward(self, x):
        x_temp = self.act_func(self.layer_input(x))
        for dense in self.layers:
            x_temp = self.act_func(dense(x_temp))
        x_temp = self.layer_output(x_temp)
        return x_temp

