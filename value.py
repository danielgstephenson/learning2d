from math import isfinite, sqrt

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import os

from generator import DataGenerator
from models import ValueModel, get_action_values
from reward import get_reward
import reward
from save import save_value_checkpoint

value_checkpoint_path = './checkpoints/value_checkpoint.pt'
value_model = ValueModel()
old_value_model = ValueModel().eval()
optimizer = torch.optim.AdamW(value_model.parameters(),lr=0.001)
horizon = 0

if os.path.exists(value_checkpoint_path):
    print('Loading Value Checkpoint...')
    value_checkpoint = torch.load(value_checkpoint_path, weights_only=False)
    print(value_checkpoint.keys())
    value_model.load_state_dict(value_checkpoint['model_state_dict'])
    old_value_model.load_state_dict(value_checkpoint['old_model_state_dict'])
    optimizer.load_state_dict(value_checkpoint['optimizer_state_dict'])
    horizon = value_checkpoint['horizon']

# discount = 0.9
# horizon = 0

lr = 0.001
for param_group in optimizer.param_groups:
    param_group['lr'] = lr

batch_size = 1000 # Reduce to 1000 if GPU memory is limited
generator = DataGenerator(batch_size)

self_noise = 0.2
smooth_loss = 10
loss_smoothing = 0.05
print('Training...')
for epoch in range(10000000):
    old_value_model.load_state_dict(value_model.state_dict())
    for batch in range(2000):
        optimizer.zero_grad()
        state, outcomes = generator.generate()
        output = value_model(state)
        target = torch.zeros_like(output)
        action_values = get_action_values(old_value_model, state, outcomes, horizon)
        action_value_mean = torch.mean(action_values,1,keepdim=True)
        action_value_max = torch.amax(action_values,1,keepdim=True)
        target = (1-self_noise)*action_value_max + self_noise*action_value_mean
        loss = F.mse_loss(output, target, reduction='mean')
        loss_item = loss.item()
        if not np.isfinite(loss_item): 
            print('non-finite loss')
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)
        optimizer.step()
        save_value_checkpoint(value_checkpoint_path, value_model, old_value_model, optimizer, horizon)
        smooth_loss = loss_smoothing*loss_item + (1-loss_smoothing)*smooth_loss
        if batch == 0: smooth_loss = 2 * loss_item
        message = ''
        message += f'Horizon: {horizon}, '
        message += f'Batch: {batch+1}, '
        message += f'RootLoss: {sqrt(loss_item):.1f}, '
        message += f'Smooth: {sqrt(smooth_loss):.1f}, '
        print(message)
    horizon += 1
