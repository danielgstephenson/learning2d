from math import sqrt
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import os

from generator import DataGenerator, get_simulation_state
from models import ActionModel, ValueModel
from checkpoint import save_action_checkpoint, save_value_checkpoint
from objective import get_life

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

lr = 0.00001
for param_group in value_optimizer.param_groups:
    param_group['lr'] = lr
for param_group in action_optimizer.param_groups:
    param_group['lr'] = lr

horizon = 0

epoch_size = 10000000
batch_size = 8000 # Reduce to 1000 if GPU memory is limited
generator = DataGenerator(batch_size, time_step = 0.1, step_count = 1)
discount = 0.99
noise = 0.2
other_noise = 0.2
rows = torch.arange(batch_size)
print('Training...')
for epoch in range(10000000):
    generator.setup_boundary()
    generator.reset()
    root_value_loss = 0
    action_value_range = 0
    for batch in range(epoch_size):
        value_optimizer.zero_grad()
        generator.reset()
        generator.generate_outcomes()
        state = generator.state
        outcomes = generator.outcomes
        value_output = value_model(state)
        with torch.no_grad():
            life0 = get_life(state).repeat_interleave(81, dim=0).reshape(-1,9,9)
            reward = -100 * (1 - life0)
            next_values = life0 * value_model(outcomes).reshape(-1,9,9)
            if horizon == 0:
                values = reward
            else:
                values = reward + discount*next_values
            row_means = torch.mean(values,2)
            row_mins = torch.amin(values,2)
            action_values = other_noise*row_means + (1-other_noise)*row_mins
            action_value_mean = torch.mean(action_values,1,keepdim=True)
            action_value_max = torch.amax(action_values,1,keepdim=True)
            action_value_min = torch.amin(action_values,1,keepdim=True)        
            action_value_range = torch.mean((action_value_max-action_value_min)).item()
            value_target = noise*action_value_mean + (1-noise)*action_value_max
        weight = torch.softmax(0.04*torch.abs(value_target),dim=0)
        value_loss = torch.sum(weight * (value_target - value_output) ** 2)
        root_value_loss = sqrt(value_loss.item())
        action_logits = action_model(state)
        action_probs = torch.softmax(action_logits,dim=1)
        disadvantage = action_value_max - action_values
        expected_disadvantage = torch.einsum('ij,ij->i',action_probs,disadvantage)
        action_loss = torch.mean(expected_disadvantage)
        random_action_loss = torch.mean(disadvantage)
        if not np.isfinite(value_loss.item()): 
            print('non-finite value loss')
            continue
        if not np.isfinite(action_loss.item()): 
            print('non-finite action loss')
            continue
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)
        value_optimizer.step()
        save_value_checkpoint(value_checkpoint_path, value_model, old_value_model, value_optimizer, horizon)
        if horizon > 0:
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=1.0)
            action_optimizer.step()
            save_action_checkpoint(action_checkpoint_path, action_model, action_optimizer)
        message = ''
        message += f'Horizon: {horizon}, '
        message += f'Batch: {batch+1}, '
        message += f'RootValueLoss: {root_value_loss:.02f}, '
        message += f'MeanValueTarget: {torch.sum(weight*value_target).item():.02f}, '
        message += f'ActionValueRange: {action_value_range:.02f}, '
        message += f'ActionGain: {torch.mean(action_loss - random_action_loss):.02f}, '
        print(message)
    horizon += 1
    old_value_model.load_state_dict(value_model.state_dict())