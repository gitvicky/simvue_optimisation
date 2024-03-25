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
# %%
#Training a Model Ensemble 
# loss_func = torch.nn.MSELoss()
loss_func = torch.nn.NLLLoss()
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
ensemble_inference = []
with torch.no_grad():
    for ii in range(num_models):
        ensemble_inference.append(model_ensemble[ii](train_x[-100:]))


distributions = torch.stack(ensemble_inference, dim=-1)

# Mean of the ensemble predictions
ensemble_means = distributions.mean(dim=-1)

# Variance of the ensemble predictions
ensemble_std = distributions.std(dim=-1)
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
