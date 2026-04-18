from math import sqrt
import numpy as np
import torch
from torch.func import vmap, grad
import os
import time

from generator import DataGenerator
from models import ActionModel, ValueModel
from checkpoint import save_checkpoint

value_checkpoint_path = './checkpoints/value_checkpoint.pt'
old_value_checkpoint_path = './checkpoints/old_value_checkpoint.pt'
action_checkpoint_path = './checkpoints/action_checkpoint.pt'
value_model = ValueModel()
old_value_model = ValueModel().eval()
action_model = ActionModel()
value_optimizer = torch.optim.AdamW(value_model.parameters(),lr=1e-3,weight_decay=1e-5)
action_optimizer = torch.optim.AdamW(action_model.parameters(),lr=1e-3,weight_decay=1e-5)
value_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    value_optimizer, 
    max_lr=1e-3, 
    total_steps=100_000, 
    pct_start=0.1,
    anneal_strategy='cos'
)
action_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    action_optimizer, 
    max_lr=1e-3, 
    total_steps=100_000, 
    pct_start=0.1,
    anneal_strategy='cos'
)
horizon = 0
batch = 0

if os.path.exists(value_checkpoint_path):
    print('Loading Value Checkpoint...')
    checkpoint = torch.load(value_checkpoint_path, weights_only=False)
    value_model.load_state_dict(checkpoint['model_state_dict'])
    value_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    value_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    batch = checkpoint['batch']
    horizon = checkpoint['horizon']

if os.path.exists(old_value_checkpoint_path):
    print('Loading Old Value Checkpoint...')
    checkpoint = torch.load(old_value_checkpoint_path, weights_only=False)
    old_value_model.load_state_dict(checkpoint['model_state_dict'])

if os.path.exists(action_checkpoint_path):
    print('Loading Action Checkpoint...')
    checkpoint = torch.load(action_checkpoint_path, weights_only=False)
    action_model.load_state_dict(checkpoint['model_state_dict'])
    action_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    action_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

for group in value_optimizer.param_groups:
    group.setdefault('initial_lr', group['lr'])
for group in action_optimizer.param_groups:
    group.setdefault('initial_lr', group['lr'])

# horizon = 0

epoch_size = 100000000000
batch_size = 4096
time_step  = 0.1
step_count = 20
discount = 0.99
discount_factor = discount ** step_count
get_per_sample_grad = vmap(grad(lambda x: value_model(x).sum()))
print('Training...')
for _ in range(10000000):
    generator = DataGenerator(old_value_model, batch_size,time_step,step_count,discount)
    for _ in range(epoch_size):
        start_time = time.perf_counter()
        value_optimizer.zero_grad()
        action_optimizer.zero_grad()
        state, value_target = generator.generate(horizon)
        value_output = value_model(state)
        mean_value_target = torch.mean(value_target)
        value_loss = torch.mean((value_target - value_output) ** 2)
        root_value_loss = sqrt(value_loss.item())
        if not np.isfinite(value_loss.item()): 
            print('non-finite value loss')
            continue
        value_loss.backward()
        value_optimizer.step()
        value_scheduler.step()
        save_checkpoint(value_checkpoint_path,value_model,value_optimizer,value_scheduler,batch,horizon)
        message = ''
        message += f'Horizon: {horizon}, '
        message += f'Batch: {batch+1}, '
        message += f'RootValueLoss: {root_value_loss:.02f}, '
        message += f'MeanValueTarget: {mean_value_target:.02f}, '
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
        action_optimizer.step()
        action_scheduler.step()
        save_checkpoint(action_checkpoint_path,action_model,action_optimizer,action_scheduler,batch,horizon)
        message += f'ActionRMSD: {action_rmsd:.04f}, '
        message += f'RootActionLoss: {root_action_loss:.04f}, '
        message += f'ActionRatio: {action_ratio:.04f}, '
        end_time = time.perf_counter()
        message += f'BatchTime: {end_time - start_time:.06f}, '
        print(message)
        batch += 1
    horizon += 1
    batch = 0
    old_value_model.load_state_dict(value_model.state_dict())