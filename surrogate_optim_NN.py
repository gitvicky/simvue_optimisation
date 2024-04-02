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

n_data = 1000
x1 = np.random.uniform(-np.pi, np.pi, n_data)
x2 = np.random.uniform(-5, 5, n_data)
x3 = np.random.uniform(-1,1,n_data)

y = func(x1, x2, x3)
# %% 
#Prepping the data for NNs
#Stacking the inputs and converting it into tensors
train_x = Tensorize(x1,x2,x3)
train_y = Tensorize(y)

#Normalising the Input Data 
input_normalizer = Normalizer(train_x)
train_x = input_normalizer.encode(train_x)

#Normalising the Ouput Data 
output_normalizer = Normalizer(train_y)
train_y = output_normalizer.encode(train_y)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=500, shuffle=True)

test_x = train_x[-100:]
test_y = train_y[-100:]
# %%
#Training a Model Ensemble 
loss_func = torch.nn.MSELoss()
model_ensemble = []
loss_ensemble = []
num_models = 5
epochs = 1000

for ii in tqdm(range(num_models), desc='Ensemble'):
    model = MLP(3, 1, 3, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_curve = []
    for epoch in tqdm(range(epochs), desc='Model Training'):
        for xx, yy in train_loader:
            optimizer.zero_grad()
            yy_nn = model(xx)
            loss = loss_func(yy[:,0], yy_nn[:,0])
            loss.backward()
            optimizer.step()
        loss_curve.append(loss.item())
    loss_ensemble.append(loss_curve)
    model_ensemble.append(model)
        
# %%
plt.figure()
for ii in range(num_models):
    plt.plot(loss_ensemble[ii], label=ii)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# %% 
#Model Inference: 
# ensemble_inference = []
# with torch.no_grad():
#     for ii in range(num_models):
#         ensemble_inference.append(model_ensemble[ii](test_x))
# distributions = torch.stack(ensemble_inference, dim=0)

# # Mean of the ensemble predictions
# ensemble_means = distributions.mean(dim=0)

# # Variance of the ensemble predictions
# ensemble_std = distributions.std(dim=0)

# %% 
#Model Inference Using vmap

from torch.func import stack_module_state
from torch.func import functional_call, jacrev, vmap

#Stack the parameters across the ensemble
params, buffers = stack_module_state(model_ensemble)

def fmodel(params, buffers, x):
    return functional_call(model_ensemble[0], (params, buffers), (x,))

predictions_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, test_x)

ensemble_means = predictions_vmap.mean(axis=0).detach().numpy()
ensemble_std = predictions_vmap.std(axis=0).detach().numpy()

# assert torch.allclose(predictions_vmap.mean(axis=0), ensemble_means, atol=1e-3, rtol=1e-5)

# %%
# Create a figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Plot ensemble means
ax.plot(ensemble_means, label='Ensemble Mean', color='tab:blue')

# Shade the area between mean and mean +/- variance
mean_plus_std = ensemble_means + ensemble_std
mean_minus_std = ensemble_means - ensemble_std
ax.fill_between(range(len(ensemble_means)), mean_minus_std[:, 0], mean_plus_std[:, 0], alpha=0.3, color='tab:blue', label='std')

# Set title, labels, and legend
ax.set_title('Ensemble Means and STD')
ax.set_xlabel('Input Sample')
ax.set_ylabel('Output')
ax.legend()

# Show the plot
plt.show()
# %%
#Obtaining the gradients
test_x.requires_grad=True
def fmodel_grads(params, x):
    return functional_call(model_ensemble[0], params, x)
    # return torch.autograd.grad(preds, x, grad_outputs=torch.ones_like(preds), create_graph=True)[0]

# preds_grads_vmap = vmap(fmodel_grads, in_dims=(0, 0, None))(params, test_x)

# %%
predictions = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, test_x)

def get_grads(y, X):
    return torch.autograd.grad(y, X, grad_outputs=torch.ones_like(y), create_graph=True)[0]

grads = []
for ii in range(num_models):
    grads.append(get_grads(predictions[ii], test_x))
grads = torch.stack(grads)

grads_means = grads.mean(axis=0).detach().numpy()
grads_std = grads.std(axis=0).detach().numpy()

# %%
#plotting the estimated gradients
def create_subplots(num_subplots, means, stds):
    # Calculate the number of rows and columns needed
    num_rows = (num_subplots + 1) // 2
    num_cols = 2

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5*num_rows))

    # Flatten the axes array if num_subplots > 1
    if num_subplots > 1:
        axes = axes.flatten()

    # Iterate over the subplots and customize them
    for ii in range(num_subplots):
        # Get the current subplot
        ax = axes[ii] if num_subplots > 1 else axes

        # Plot ensemble means
        ax.plot(means[:,ii], label='Gradients', color='tab:blue')

        # Shade the area between mean and mean +/- variance
        mean_plus_std = means + stds
        mean_minus_std = means - stds
        ax.fill_between(range(len(means)), mean_minus_std[:, ii], mean_plus_std[:, ii], alpha=0.3, color='tab:blue', label='std')

        # Customize the subplot (e.g., add title, labels, plot data)
        ax.set_title(f"Variable {ii+1}")
        ax.set_xlabel(f"x{ii+1}")
        ax.set_ylabel(f"dy/dx{ii+1}")
        # Add your plotting code here

    # Remove unused subplots if num_subplots is odd
    if num_subplots % 2 != 0:
        fig.delaxes(axes[-1])

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()


create_subplots(3, grads_means, grads_std)
# %%
