
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.func import vmap, grad
import os
import time

from generator import DataGenerator
from value import ValueModel
from checkpoint import save_checkpoint

checkpoint_path = './checkpoints/checkpoint.pt'
old_checkpoint_path = './checkpoints/old_checkpoint.pt'
value_model = ValueModel()
old_value_model = ValueModel().eval()
value_optimizer = torch.optim.AdamW(value_model.parameters(),lr=1e-3)
horizon = 0
batch = 0

if os.path.exists(checkpoint_path):
    print(f'Loading Checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    value_model.load_state_dict(checkpoint['model_state_dict'])
    old_value_model.load_state_dict(checkpoint['model_state_dict'])
    value_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    batch = checkpoint['batch']
    horizon = checkpoint['horizon']

if os.path.exists(old_checkpoint_path):
    print(f'Loading Old Checkpoint from {old_checkpoint_path}...')
    checkpoint = torch.load(old_checkpoint_path, weights_only=False)
    old_value_model.load_state_dict(checkpoint['model_state_dict'])
else:
    old_value_model.load_state_dict(value_model.state_dict())
    save_checkpoint(old_checkpoint_path, value_model, value_optimizer, batch, horizon)

for param_group in value_optimizer.param_groups:
    param_group['lr'] = 1e-4

# horizon = 0
# batch = 0

sim_count = 5000
batch_count = 10
epoch_count = 1
step_count = 1
minibatch_size = 500
time_step = 0.1
minibatch_count = sim_count // minibatch_size
print('minibatch_count',minibatch_count)
quality_history = []
cuda_generator = torch.Generator(device='cuda')
data_generator = DataGenerator(old_value_model, sim_count, step_count, time_step)
last_log_time = time.perf_counter()
quality_threshold = 0.97
quality = 0

print('Training...')
for _ in range(100000000):
    start_time = time.perf_counter()
    full_state, full_target = data_generator.generate(horizon)
    dataset = TensorDataset(full_state, full_target)
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True, generator=cuda_generator)
    for epoch in range(epoch_count):
        for state, target in loader:
            value_optimizer.zero_grad()
            estimate = value_model(state)
            value_loss = F.mse_loss(estimate, target)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)
            value_optimizer.step()
        with torch.no_grad():
            full_estimate = value_model(full_state)
            full_loss = F.mse_loss(full_estimate, full_target)
            null_estimate = 0*full_target + full_target.mean()
            null_loss = F.mse_loss(null_estimate, full_target)
            quality = 1 - full_loss/null_loss
            quality_history.append(quality.item())
            message = ''
            message += f'Horizon: {horizon}, '
            message += f'Batch: {batch+1}, '
            message += f'Epoch: {epoch+1}, '
            message += f'ModelQuality: {quality:.03f}, '
            message += f'TargetMean: {torch.mean(full_target):.03f}, '
            message += f'TargetSD: {sqrt(null_loss):.03f}, '
            now = time.perf_counter()
            message += f'Time: {now - last_log_time:.03f}, '
            last_log_time = now
            print(message)
    save_checkpoint(checkpoint_path, value_model, value_optimizer, batch, horizon)
    if batch + 1 >= batch_count and quality > quality_threshold:
        print(f'Horizon {horizon} Complete.')
        horizon += 1
        batch = 0
        quality_history = []
        print(f'Saving checkpoint...')
        old_value_model.load_state_dict(value_model.state_dict())
        save_checkpoint(old_checkpoint_path, value_model, value_optimizer, batch, horizon)
        print(f'Beginning Horizon {horizon}...')
        continue
    batch += 1