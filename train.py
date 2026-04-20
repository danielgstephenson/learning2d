from math import sqrt
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.func import vmap, grad
import os
import time

from physics import active_action_tensor
from generator import DataGenerator
from models import ActionModel, ValueModel
from checkpoint import save_checkpoint

value_checkpoint_path = './checkpoints/value_checkpoint.pt'
old_value_checkpoint_path = './checkpoints/value_checkpoint0.pt'
action_checkpoint_path = './checkpoints/action_checkpoint.pt'
value_model = ValueModel()
old_value_model = ValueModel().eval()
action_model = ActionModel()
value_optimizer = torch.optim.AdamW(value_model.parameters(),lr=1e-3)
action_optimizer = torch.optim.AdamW(action_model.parameters(),lr=1e-3)
horizon = 0
batch = 0

if os.path.exists(value_checkpoint_path):
    print('Loading Value Checkpoint...')
    checkpoint = torch.load(value_checkpoint_path, weights_only=False)
    value_model.load_state_dict(checkpoint['model_state_dict'])
    value_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    batch = checkpoint['batch']
    horizon = checkpoint['horizon']

if os.path.exists(action_checkpoint_path):
    print('Loading Action Checkpoint...')
    checkpoint = torch.load(action_checkpoint_path, weights_only=False)
    action_model.load_state_dict(checkpoint['model_state_dict'])
    action_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if os.path.exists(old_value_checkpoint_path):
    print('Loading Old Value Checkpoint...')
    checkpoint = torch.load(old_value_checkpoint_path, weights_only=False)
    old_value_model.load_state_dict(checkpoint['model_state_dict'])
else: 
    save_checkpoint(old_value_checkpoint_path,value_model,value_optimizer,batch,horizon)

horizon = 1
batch = 0

for param_group in value_optimizer.param_groups:
    param_group['lr'] = 1e-4
for param_group in action_optimizer.param_groups:
    param_group['lr'] = 1e-4

batch_size = 4096
get_costate = vmap(grad(lambda x: value_model(x).sum()))
generator = DataGenerator(old_value_model, batch_size)

print('Training...')
for _ in range(100000000):
    start_time = time.perf_counter()
    value_optimizer.zero_grad()
    action_optimizer.zero_grad()
    state, value_target = generator.generate(horizon)
    value_logits = value_model(state)
    value_loss = F.binary_cross_entropy_with_logits(value_logits, value_target)
    if np.isfinite(value_loss.item()): 
        value_loss.backward()
        value_optimizer.step()
    else:
        print('non-finite value loss')
        continue
    action_logits = action_model(state)
    costate = get_costate(state)
    velocity_gradient = costate[:,[8,9]]
    action_values = torch.einsum('ij,kj->ik',velocity_gradient,active_action_tensor)
    action_target = torch.argmax(action_values, dim=1)
    action_loss = F.cross_entropy(action_logits,action_target)
    if np.isfinite(action_loss.item()): 
        action_loss.backward()
        action_optimizer.step()
    else:
        print('non-finite action loss')
        continue
    if (batch + 1) % 10 == 0 or batch == 0:
        with torch.no_grad():
            gradient_ratio = action_loss / (torch.var(velocity_gradient) + 1e-8)
            null_value_probs = 0*value_target + value_target.mean()
            null_value_loss = F.binary_cross_entropy(null_value_probs, value_target)
            value_R2 = 1 - value_loss/null_value_loss
            action_counts = torch.bincount(action_target, minlength=8).float()
            null_action_probs = action_counts / action_counts.sum()
            null_action_logits = torch.log(null_action_probs + 1e-12)
            null_action_batch = null_action_logits.unsqueeze(0).expand(batch_size, -1)
            null_action_loss = F.cross_entropy(null_action_batch, action_target)
            action_R2 = 1 - action_loss/null_action_loss
        message = ''
        message += f'Horizon: {horizon}, '
        message += f'Batch: {batch+1}, '
        message += f'ValR2: {value_R2:.03f}, '
        message += f'ActR2: {action_R2:.03f}, '
        stop_time = time.perf_counter()
        message += f'Time: {stop_time-start_time:.03f}, '
        print(message)
    if batch % 100 == 0:
        save_checkpoint(value_checkpoint_path,value_model,value_optimizer,batch,horizon)
        save_checkpoint(action_checkpoint_path,action_model,action_optimizer,batch,horizon)
    batch += 1