
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

sim_count = 2000
step_count = 50
batch_size = sim_count*step_count
batch_count = 20
minibatch_size = 4000
minibatch_count = batch_size // minibatch_size
print('minibatch_count',minibatch_count)
epochs = 2
cuda_generator = torch.Generator(device='cuda')
data_generator = DataGenerator(old_value_model, sim_count, step_count)
last_log_time = time.perf_counter()

print('Training...')
for _ in range(100000000):
    start_time = time.perf_counter()
    full_state, full_target = data_generator.generate(horizon)
    dataset = TensorDataset(full_state, full_target)
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True, generator=cuda_generator)
    for epoch in range(epochs):
        for state, target in loader:
            value_optimizer.zero_grad()
            logits = value_model(state)
            value_loss = F.binary_cross_entropy_with_logits(logits, target)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)
            value_optimizer.step()
    with torch.no_grad():
        full_logits = value_model(full_state)
        full_loss = F.binary_cross_entropy_with_logits(full_logits, full_target)
        null_probs = 0*full_target + full_target.mean()
        null_loss = F.binary_cross_entropy(null_probs, full_target)
        quality = 1 - full_loss/null_loss
    message = ''
    message += f'Horizon: {horizon}, '
    message += f'Batch: {batch+1}, '
    message += f'ModelQuality: {quality:.03f}, '
    message += f'MeanTarget: {torch.mean(full_target):.03f}, '
    now = time.perf_counter()
    message += f'Time: {now - last_log_time:.03f}, '
    last_log_time = now
    print(message)
    save_checkpoint(checkpoint_path, value_model, value_optimizer, batch, horizon)
    if batch + 1 >= batch_count:
        print(f'Horizon {horizon} Complete.')
        old_value_model.load_state_dict(value_model.state_dict())
        horizon += 1
        batch = 0
        print(f'Saving checkpoint...')
        old_value_model.load_state_dict(value_model.state_dict())
        save_checkpoint(old_checkpoint_path, value_model, value_optimizer, batch, horizon)
        print(f'Beginning Horizon {horizon}...')
        continue
    batch += 1