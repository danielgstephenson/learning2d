from math import isfinite

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from generator import DataGenerator
import os

from reward import get_reward
from save import save_action_value_checkpoint
from models import ValueModel, get_action_values, ActionModel
    
value_checkpoint_path = './checkpoints/value_checkpoint.pt'
action_checkpoint_path = './checkpoints/action_checkpoint.pt'
value_model = ValueModel().eval()
action_model = ActionModel()
optimizer = torch.optim.AdamW(action_model.parameters(),lr=0.001)
discount = 0.9
horizon = 0

if os.path.exists(value_checkpoint_path):
    print('Loading Value Checkpoint...')
    checkpoint = torch.load(value_checkpoint_path, weights_only=False)
    value_model.load_state_dict(checkpoint['model_state_dict'])
    horizon = checkpoint['horizon']

if os.path.exists(action_checkpoint_path):
    print('Loading Action Checkpoint...')
    checkpoint = torch.load(action_checkpoint_path, weights_only=False)
    action_model.load_state_dict(checkpoint['model_state_dict'])

# discount = 0.9
# horizon = 0

lr = 0.001
for param_group in optimizer.param_groups:
    param_group['lr'] = lr

batch_size = 1000 # Reduce to 1000 if GPU memory is limited
generator = DataGenerator(batch_size)

smooth_loss = 10
loss_smoothing = 0.05
print('Training...')
for batch in range(10000000):
    optimizer.zero_grad()
    state, outcomes = generator.generate()
    action_probs = action_model(state)
    action_values = get_action_values(value_model, state, outcomes, horizon)
    best_actions = torch.argmax(action_values, dim=1)
    action_value_max = torch.amax(action_values,1,keepdim=True)
    action_value_expected = torch.sum(action_probs*action_values,1,keepdim=True)
    mistake = action_value_max - action_value_expected
    loss = torch.mean(mistake)
    loss_item = loss.item()
    if not np.isfinite(loss_item): 
        print('non-finite loss')
        continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=1.0)
    optimizer.step()
    save_action_value_checkpoint(action_checkpoint_path, action_model, optimizer)
    smooth_loss = loss_smoothing*loss_item + (1-loss_smoothing)*smooth_loss
    if batch == 0: smooth_loss = 2 * loss_item
    message = ''
    message += f'Batch: {batch+1}, '
    message += f'Discount: {discount}, '
    message += f'Loss: {loss_item:.4f}, '
    message += f'Smooth: {smooth_loss:.4f}, '
    # print(message)
    x = torch.stack((
        torch.argmax(action_values,dim=1)[0:10],
        torch.argmax(action_probs,dim=1)[0:10]
    ))
    print(x)
    # print(torch.argmax(action_probs,dim=1)[0:10])