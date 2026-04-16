from math import sqrt
import numpy as np
import torch
from torch.func import vmap, grad
import torch.nn.functional as F
import os

from generator import DataGenerator
from models import ActionModel, ValueModel
from checkpoint import save_action_checkpoint, save_value_checkpoint

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
    checkpoint = torch.load(value_checkpoint_path, weights_only=False)
    value_model.load_state_dict(checkpoint['model_state_dict'])
    old_value_model.load_state_dict(checkpoint['old_model_state_dict'])
    value_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    horizon = checkpoint['horizon']

if os.path.exists(action_checkpoint_path):
    print('Loading Action Checkpoint...')
    checkpoint = torch.load(action_checkpoint_path, weights_only=False)
    action_model.load_state_dict(checkpoint['model_state_dict'])
    action_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

lr = 0.0001
for param_group in value_optimizer.param_groups:
    param_group['lr'] = lr
for param_group in action_optimizer.param_groups:
    param_group['lr'] = lr

# horizon = 0

epoch_size = 100000000000
batch_size = 64000 # Reduce to 1000 if GPU memory is limited
time_step  = 0.1
step_count = 20
discount = 0.99
discount_factor = discount ** step_count
generator = DataGenerator(batch_size,time_step,step_count,discount)
get_per_sample_grad = vmap(grad(lambda x: value_model(x).sum()))
print('Training...')
for epoch in range(10000000):
    for batch in range(epoch_size):
        value_optimizer.zero_grad()
        state, value_target = generator.generate(old_value_model, horizon)
        value_output = value_model(state)
        weight = torch.softmax(-0.01*value_target,dim=0)
        mean_value_target = torch.sum(weight*value_target)
        value_loss = torch.sum(weight * (value_target - value_output) ** 2)
        root_value_loss = sqrt(value_loss.item())
        if not np.isfinite(value_loss.item()): 
            print('non-finite value loss')
            continue
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)
        value_optimizer.step()
        save_value_checkpoint(value_checkpoint_path, value_model, old_value_model, value_optimizer, horizon)
        action_output = action_model(state)
        costate = get_per_sample_grad(state)
        velocity_gradient = costate[:,[8,9]]
        action_loss = torch.mean((velocity_gradient - action_output) ** 2)
        root_action_loss = torch.sqrt(action_loss)
        action_mean = torch.mean(velocity_gradient, dim=0, keepdim=True)
        action_rmsd = torch.sqrt(torch.mean((velocity_gradient - action_mean) ** 2))
        action_ratio = root_action_loss / action_rmsd
        if not np.isfinite(action_loss.item()): 
            print('non-finite action loss')
            continue
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=1.0)
        action_optimizer.step()
        save_action_checkpoint(action_checkpoint_path, action_model, action_optimizer)
        message = ''
        message += f'Horizon: {horizon}, '
        message += f'Batch: {batch+1}, '
        message += f'RootValueLoss: {root_value_loss:.02f}, '
        message += f'MeanValueTarget: {mean_value_target:.02f}, '
        message += f'ActionRMSD: {action_rmsd:.04f}, '
        message += f'RootActionLoss: {root_action_loss:.04f}, '
        message += f'ActionRatio: {action_ratio:.04f}, '
        print(message)
    horizon += 1
    old_value_model.load_state_dict(value_model.state_dict())