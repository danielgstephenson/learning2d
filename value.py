from math import isfinite

import numpy as np
import torch
import torch.nn.functional as F
from generator import DataGenerator
from model import ValueModel
from physics import device
import os

from reward import get_life, get_reward
from save import save_checkpoint

checkpoint_path = './checkpoints/value_checkpoint.pt'
model = ValueModel()
old_model = ValueModel().eval()
optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
discount = 0.9
horizon = 0

if os.path.exists(checkpoint_path):
    print('Loading Checkpoint...')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    old_model.load_state_dict(checkpoint['old_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    discount = checkpoint['discount']
    horizon = checkpoint['horizon']

# discount = 0.9
# horizon = 0

lr = 0.0001
for param_group in optimizer.param_groups:
    param_group['lr'] = lr

batch_size = 1000 # Reduce to 1000 if GPU memory is limited
generator = DataGenerator(batch_size)

noise = 0.2
other_noise = 0.7
smooth_loss = 0
loss_smoothing = 0.05
loss_threshold = 0.02
print('Training...')
optimizer.zero_grad()
for batch in range(100000000000):
    optimizer.zero_grad()
    state, outcomes = generator.generate()
    output = model(state)
    target = torch.zeros_like(output)
    with torch.no_grad():
        reward = get_reward(state).reshape(batch_size,1,1)
        life = get_life(state).reshape(batch_size,1,1)
        next_values = life * get_reward(outcomes).reshape((batch_size,9,9)) # Should use old_model if horizon > 0
        values = (1-discount)*reward + discount*next_values
        row_means = torch.mean(values,2)
        row_mins = torch.amin(values,2)
        action_values = (1-other_noise)*row_mins + other_noise*row_means
        action_value_mean = torch.mean(action_values,1)
        action_value_max = torch.amax(action_values,1)
        target = (1-noise)*action_value_max + noise*action_value_mean
        target = target.unsqueeze(1)
    loss = F.mse_loss(output, target, reduction='mean')
    loss_value = loss.detach().cpu().numpy()
    if not np.isfinite(loss_value): continue
    smooth_loss = loss_smoothing*loss_value + (1-loss_smoothing)*smooth_loss
    if batch == 0: smooth_loss = 2 * loss_value
    message = ''
    message += f'Batch: {batch+1}, '
    message += f'Discount: {discount}, '
    message += f'Horizon: {horizon}, '
    message += f'Loss: {loss_value:07.2f}, '
    message += f'Smooth: {smooth_loss:07.2f}, '
    print(message)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    save_checkpoint(model, old_model, optimizer, discount, horizon, checkpoint_path)
