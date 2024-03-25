#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23rd March, 2024

Building the base functionality for min-max optimisation using surrogate modelling 
"""
# %% 
#Importing the necessary packages. 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from tqdm import tqdm 
import torch 
import gpytorch 

from utils import * 

# %% 
#Arbitrary multi-input single-output function definition. 

def func(x1, x2, x3):

    y =  np.sin(x1*x2) + x3
    return y 

n_data = 10000
x1 = np.random.uniform(-np.pi, np.pi, n_data)
x2 = np.random.uniform(-5, 5, n_data)
x3 = np.random.uniform(-1,1,n_data)

y = func(x1, x2, x3)
# %% 
#Prepping the data for GPs. 
#Stacking the inputs and converting it into tensors
train_x = Tensorize(x1,x2,x3)
train_y = Tensorize(y).squeeze()

#Normalising the Input Data 
input_normalizer = Standardizer(train_x)
train_x= input_normalizer.encode(train_x)

#Normalising the Ouput Data 
output_normalizer = Standardizer(train_y)
train_y = output_normalizer.encode(train_y)
# %%
#Setting up a GP as a surrogate model (GpyTorch)
means = gpytorch.means.ConstantMean()
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = means
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)


# %%
#Training the Surroagte Model. 
epochs = 50

# Switching to train
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes the Models and the Likelihood parameters. 

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for epoch in range(epochs):
    optimizer.zero_grad()
    gp_output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(gp_output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        epoch + 1, epochs, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item() 
    ))
    optimizer.step()
# %%  
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

test_x = train_x[-100:]
# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    # ax.plot(test_x.numpy(), train_u[-100:,0].numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(np.arange(0, 100), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(np.arange(0,100), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
# %%
import numpy as np

def generate_hypercube_grid(lower, upper, num_points):
    """
    Generate an equally spaced hypercube of grid points with 3 dimensions.
    
    Parameters:
    - start: The starting value for each dimension (inclusive).
    - end: The ending value for each dimension (inclusive).
    - num_points: The number of grid points along each dimension.
    
    Returns:
    - grid: A 2D numpy array representing the hypercube grid points,
            with each row indicating the point index and the three values.
    """
    # Generate the coordinate values for each dimension
    x = np.linspace(lower[0], upper[0], num_points)
    y = np.linspace(lower[1], upper[1], num_points)
    z = np.linspace(lower[2], upper[2], num_points)
    
    # Create a meshgrid from the coordinate values
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Flatten the meshgrid arrays
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    return Tensorize(X_flat, Y_flat, Z_flat)

lb = [-5, -5, -5] # Lower Bound 
ub = [5, 5, 5] #Upper Bound
num_points = 10 #Total Evaluation Points -- num_points**3

test_x = generate_hypercube_grid(lb, ub, num_points)
test_x = input_normalizer.encode(test_x)

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(test_x))
    pred_mean = pred.mean
    pred_std = pred.stddev

pred_mean = output_normalizer.decode(pred_mean)
pred_std = output_normalizer.decode(pred_std)
# %%
