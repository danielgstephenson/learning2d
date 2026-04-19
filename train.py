from math import sqrt
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.func import vmap, grad
import os
import time

from generator import DataGenerator
from models import GradientModel, ValueModel, discretize
from checkpoint import save_checkpoint

value_checkpoint_path = './checkpoints/value_checkpoint.pt'
old_value_checkpoint_path = './checkpoints/old_value_checkpoint.pt'
gradient_checkpoint_path = './checkpoints/gradient_checkpoint.pt'
value_model = ValueModel()
old_value_model = ValueModel().eval()
gradient_model = GradientModel()
value_optimizer = torch.optim.AdamW(value_model.parameters(),lr=1e-3,weight_decay=1e-5)
gradient_optimizer = torch.optim.AdamW(gradient_model.parameters(),lr=1e-3,weight_decay=1e-5)
max_learning_rate = 1e-04
value_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    value_optimizer, 
    max_lr=max_learning_rate, 
    total_steps=100_000, 
    pct_start=0.1,
    anneal_strategy='cos'
)
gradient_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    gradient_optimizer, 
    max_lr=max_learning_rate, 
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
max_learning_rate = 1e-04
value_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    value_optimizer, 
    max_lr=max_learning_rate, 
    total_steps=100_000, 
    pct_start=0.1,
    anneal_strategy='cos'
)
gradient_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    gradient_optimizer, 
    max_lr=max_learning_rate, 
    total_steps=100_000, 
    pct_start=0.1,
    anneal_strategy='cos'
)

batch_size = 4096
top_k_val = int(batch_size * 0.1)
time_step  = 0.1
step_count = 20
discount = 0.99
discount_factor = discount ** step_count
get_per_sample_grad = vmap(grad(lambda x: value_model.get_expected_value(x).sum()))
generator = DataGenerator(old_value_model, batch_size,time_step,step_count,discount)

print('Training...')
for _ in range(100000000):
    start_time = time.perf_counter()
    value_optimizer.zero_grad()
    gradient_optimizer.zero_grad()
    state, value_target = generator.generate(horizon)
    value_logits = value_model(state)
    value_loss = F.cross_entropy(value_logits, value_target)
    if np.isfinite(value_loss.item()): 
        value_loss.backward()
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
        with torch.no_grad():
            probs = torch.softmax(value_logits, dim=1)
            target_probs = torch.gather(probs, 1, value_target.unsqueeze(1))
            value_accuracy = target_probs.mean().item()
            mean_value_output = value_model.get_expected_value(state)
            mean_value_target = value_model.midpoints[value_target]
            value_mae = torch.mean(torch.abs(mean_value_target - mean_value_output)).item()
            gradient_ratio = gradient_loss / (torch.var(velocity_gradient) + 1e-8)
        message = ''
        message += f'Horizon: {horizon}, '
        message += f'Batch: {batch+1}, '
        message += f'ValLoss: {value_loss:.02f}, '
        message += f'ValAcc: {value_accuracy:.02f}, '
        message += f'ValMAE: {value_mae:.02f}, '
        message += f'GradRatio: {gradient_ratio:.02f}, '
        stop_time = time.perf_counter()
        message += f'Time: {stop_time-start_time:.02f}, '
        print(message)
    if batch % 100 == 0:
        save_checkpoint(value_checkpoint_path,value_model,value_optimizer,value_scheduler,batch,horizon)
        save_checkpoint(gradient_checkpoint_path,gradient_model,gradient_optimizer,gradient_scheduler,batch,horizon)
    batch += 1