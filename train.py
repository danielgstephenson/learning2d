from math import isfinite, sqrt

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import os

from generator import DataGenerator
from models import ActionModel, ValueModel, get_action_values
from reward import get_reward
import reward
from save import save_action_checkpoint, save_value_checkpoint

value_checkpoint_path = './checkpoints/value_checkpoint.pt'
action_checkpoint_path = './checkpoints/action_checkpoint.pt'
value_model = ValueModel()
old_value_model = ValueModel().eval()
action_model = ActionModel()
value_optimizer = torch.optim.AdamW(value_model.parameters(),lr=0.001)
action_optimizer = torch.optim.AdamW(action_model.parameters(),lr=0.001)
horizon = 0

if os.path.exists(value_checkpoint_path):
    print('Loading Value Checkpoint...')
    value_checkpoint = torch.load(value_checkpoint_path, weights_only=False)
    print(value_checkpoint.keys())
    value_model.load_state_dict(value_checkpoint['model_state_dict'])
    old_value_model.load_state_dict(value_checkpoint['old_model_state_dict'])
    value_optimizer.load_state_dict(value_checkpoint['optimizer_state_dict'])
    horizon = value_checkpoint['horizon']

if os.path.exists(action_checkpoint_path):
    print('Loading Action Checkpoint...')
    checkpoint = torch.load(action_checkpoint_path, weights_only=False)
    action_model.load_state_dict(checkpoint['model_state_dict'])

# discount = 0.9
# horizon = 0

lr = 0.001
for param_group in value_optimizer.param_groups:
    param_group['lr'] = lr

batch_size = 1000 # Reduce to 1000 if GPU memory is limited
generator = DataGenerator(batch_size)

self_noise = 0.2
print('Training...')
for epoch in range(10000000):
    old_value_model.load_state_dict(value_model.state_dict())
    for batch in range(2000):
        value_optimizer.zero_grad()
        state, outcomes = generator.generate()
        action_output = action_model(state)
        value_output = value_model(state)
        value_target = torch.zeros_like(value_output)
        action_values = get_action_values(old_value_model, state, outcomes, horizon)
        best_actions = torch.argmax(action_values, dim=1)
        action_loss = F.mse_loss(action_output, action_values, reduction='mean')
        action_loss_item = action_loss.item()
        action_value_mean = torch.mean(action_values,1,keepdim=True)
        action_value_max = torch.amax(action_values,1,keepdim=True)
        value_target = (1-self_noise)*action_value_max + self_noise*action_value_mean
        value_loss = F.mse_loss(value_output, value_target, reduction='mean')
        value_loss_item = value_loss.item()
        if not np.isfinite(value_loss_item): 
            print('non-finite value loss')
            continue
        if not np.isfinite(action_loss_item): 
            print('non-finite action loss')
            continue
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)
        value_optimizer.step()
        save_value_checkpoint(value_checkpoint_path, value_model, old_value_model, value_optimizer, horizon)
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=1.0)
        value_optimizer.step()
        save_action_checkpoint(action_checkpoint_path, action_model, action_optimizer)
        message = ''
        message += f'Horizon: {horizon}, '
        message += f'Batch: {batch+1}, '
        message += f'RootValueLoss: {sqrt(value_loss_item):.1f}, '
        message += f'RootActionLoss: {sqrt(action_loss_item):.2f}, '
        print(message)
    horizon += 1
