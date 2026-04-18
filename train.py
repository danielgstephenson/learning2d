from math import sqrt
import numpy as np
import torch
from torch.func import vmap, grad
import os
import time

from generator import DataGenerator
from models import GradientModel, ValueModel
from checkpoint import save_checkpoint

value_checkpoint_path = './checkpoints/value_checkpoint.pt'
old_value_checkpoint_path = './checkpoints/old_value_checkpoint.pt'
gradient_checkpoint_path = './checkpoints/gradient_checkpoint.pt'
value_model = ValueModel()
old_value_model = ValueModel().eval()
gradient_model = GradientModel()
value_optimizer = torch.optim.AdamW(value_model.parameters(),lr=1e-3,weight_decay=1e-5)
gradient_optimizer = torch.optim.AdamW(gradient_model.parameters(),lr=1e-3,weight_decay=1e-5)
value_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    value_optimizer, 
    max_lr=1e-3, 
    total_steps=100_000, 
    pct_start=0.1,
    anneal_strategy='cos'
)
gradient_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    gradient_optimizer, 
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

if os.path.exists(gradient_checkpoint_path):
    print('Loading Action Checkpoint...')
    checkpoint = torch.load(gradient_checkpoint_path, weights_only=False)
    gradient_model.load_state_dict(checkpoint['model_state_dict'])
    gradient_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    gradient_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

if os.path.exists(old_value_checkpoint_path):
    print('Loading Old Value Checkpoint...')
    checkpoint = torch.load(old_value_checkpoint_path, weights_only=False)
    old_value_model.load_state_dict(checkpoint['model_state_dict'])
else: 
    save_checkpoint(old_value_checkpoint_path,value_model,value_optimizer,value_scheduler,batch,horizon)

for group in value_optimizer.param_groups:
    group.setdefault('initial_lr', group['lr'])
for group in gradient_optimizer.param_groups:
    group.setdefault('initial_lr', group['lr'])

horizon = 0
batch = 0
value_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    value_optimizer, 
    max_lr=1e-3, 
    total_steps=100_000, 
    pct_start=0.1,
    anneal_strategy='cos'
)
gradient_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    gradient_optimizer, 
    max_lr=1e-3, 
    total_steps=100_000, 
    pct_start=0.1,
    anneal_strategy='cos'
)

epoch_size = 100000000000
batch_size = 4096
time_step  = 0.1
step_count = 20
discount = 0.99
discount_factor = discount ** step_count
get_per_sample_grad = vmap(grad(lambda x: value_model(x).sum()))
generator = DataGenerator(old_value_model, batch_size,time_step,step_count,discount)
print('Training...')
for _ in range(epoch_size):
    start_time = time.perf_counter()
    value_optimizer.zero_grad()
    gradient_optimizer.zero_grad()
    state, value_target = generator.generate(horizon)
    value_output = value_model(state)
    value_squared_error = (value_target - value_output) ** 2
    raw_value_loss = torch.mean(value_squared_error)
    with torch.no_grad():
        value_weights = 1 + torch.abs(value_target-value_output)
        value_weights = value_weights / value_weights.sum()
    weighted_value_loss = torch.sum(value_weights * value_squared_error)
    if np.isfinite(raw_value_loss.item()): 
        weighted_value_loss.backward()
        value_optimizer.step()
        value_scheduler.step()
    else:
        print('non-finite value loss')
        continue
    gradient_output = gradient_model(state)
    costate = get_per_sample_grad(state)
    velocity_gradient = costate[:,[8,9]]
    gradient_loss = torch.mean((velocity_gradient - gradient_output) ** 2)
    if np.isfinite(gradient_loss.item()): 
        gradient_loss.backward()
        gradient_optimizer.step()
        gradient_scheduler.step()
    else:
        print('non-finite action loss')
        continue
    if (batch + 1) % 10 == 0 or batch == 0:
        root_value_loss = sqrt(raw_value_loss.item())
        max_value_err = torch.max(torch.abs(value_target - value_output)).item()
        root_weighted_value_loss = sqrt(weighted_value_loss.item())
        value_ratio = root_value_loss / (torch.std(value_target) + 1e-8)
        hit_percentage = (value_target < 0).float().mean() * 100
        root_gradient_loss = torch.sqrt(gradient_loss)
        gradient_ratio = root_gradient_loss / (torch.std(velocity_gradient) + 1e-8)
        message = ''
        message += f'Horizon: {horizon}, '
        message += f'Batch: {batch+1}, '
        message += f'RootValLoss: {root_value_loss:.02f}, '
        message += f'MaxValErr: {max_value_err:.02f}, '
        message += f'RootWtValLoss: {root_weighted_value_loss:.02f}, '
        message += f'ValRatio: {value_ratio:.02f}, '
        message += f'RootGradientLoss: {root_gradient_loss:.04f}, '
        message += f'GradientRatio: {gradient_ratio:.04f}, '
        end_time = time.perf_counter()
        message += f'BatchTime: {end_time - start_time:.04f}, '
        print(message)
    if batch % 100 == 0:
        save_checkpoint(value_checkpoint_path,value_model,value_optimizer,value_scheduler,batch,horizon)
        save_checkpoint(gradient_checkpoint_path,gradient_model,gradient_optimizer,gradient_scheduler,batch,horizon)
    batch += 1