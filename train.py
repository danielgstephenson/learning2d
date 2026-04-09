from math import sqrt
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import os

from generator import DataGenerator
from models import ActionModel, ValueModel
from checkpoint import save_action_checkpoint, save_value_checkpoint
from objective import get_action_values

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
    value_model.load_state_dict(value_checkpoint['model_state_dict'])
    old_value_model.load_state_dict(value_checkpoint['old_model_state_dict'])
    value_optimizer.load_state_dict(value_checkpoint['optimizer_state_dict'])
    horizon = value_checkpoint['horizon']
if os.path.exists(action_checkpoint_path):
    print('Loading Action Checkpoint...')
    checkpoint = torch.load(action_checkpoint_path, weights_only=False)
    action_model.load_state_dict(checkpoint['model_state_dict'])

lr = 0.001
for param_group in value_optimizer.param_groups:
    param_group['lr'] = lr

# horizon = 0

epoch_size = 10
batch_size = 5000 # Reduce to 1000 if GPU memory is limited
generator = DataGenerator(batch_size)
self_noise = 0.5
mean_value_loss = 0
print('Training...')
for epoch in range(10000000):
    for batch in range(epoch_size):
        value_optimizer.zero_grad()
        action_optimizer.zero_grad()
        generator.reset()
        generator.generate_outcomes()
        state = generator.state
        outcomes = generator.outcomes
        value_output = value_model(state)
        action_values = get_action_values(old_value_model, state, outcomes, horizon)
        action_value_mean = torch.mean(action_values,1,keepdim=True)
        action_value_max = torch.amax(action_values,1,keepdim=True)
        action_value_min = torch.amin(action_values,1,keepdim=True)
        value_target = (1-self_noise)*action_value_max + self_noise*action_value_mean
        value_loss = F.mse_loss(value_output, value_target, reduction='mean')
        best_action = torch.argmax(action_values,dim=1)
        action_logits = action_model(state)
        action_loss = F.cross_entropy(action_logits, best_action)
        chosen_action = torch.argmax(action_logits, dim=1)
        action_accuracy = torch.mean((best_action==chosen_action).to(torch.float))
        action_value_range = torch.mean((action_value_max-action_value_min))
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
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=1.0)
        action_optimizer.step()
        save_action_checkpoint(action_checkpoint_path, action_model, action_optimizer)
        message = ''
        message += f'Horizon: {horizon}, '
        message += f'Batch: {batch+1}, '
        message += f'RootValueLoss: {sqrt(value_loss.item()):.2f}, '
        message += f'ActionValueRange: {action_value_range:.2f}, '
        message += f'ActionAccuracy: {action_accuracy:.2f}, '
        print(message)
    old_value_model.load_state_dict(value_model.state_dict())
    horizon += 1