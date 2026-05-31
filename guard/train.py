
import sys
from typing import Any
from math import sqrt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.func import vmap, grad
import os
import time

from generator import DataGenerator
from models import ActionModel, ValueModel

sys.stdout = open('train.log', 'w', buffering=1)

checkpoint_path = './checkpoints/checkpoint.pt'
value_model_a = ValueModel()
value_model_b = ValueModel()
old_value_model_a = ValueModel().eval()
old_value_model_b = ValueModel().eval()
action0_model = ActionModel()
action1_model = ActionModel()
value_optimizer_a = torch.optim.AdamW(value_model_a.parameters(),lr=1e-4)
value_optimizer_b = torch.optim.AdamW(value_model_b.parameters(),lr=1e-4)
action0_optimizer = torch.optim.AdamW(action0_model.parameters(),lr=1e-4)
action1_optimizer = torch.optim.AdamW(action1_model.parameters(),lr=1e-4)
horizon = 0
batch = 0

def save_checkpoint():
    checkpoint: dict[str, Any] = { 
        'value_model_a': value_model_a.state_dict(),
        'value_model_b': value_model_b.state_dict(),
        'old_value_model_a': old_value_model_a.state_dict(),
        'old_value_model_b': old_value_model_b.state_dict(),
        'action0_model': action0_model.state_dict(),
        'action1_model': action1_model.state_dict(),
        'value_optimizer_a': value_optimizer_a.state_dict(),
        'value_optimizer_b': value_optimizer_b.state_dict(),
        'action0_optimizer': action0_optimizer.state_dict(),
        'action1_optimizer': action1_optimizer.state_dict(),
        'batch': batch,
        'horizon': horizon,
    }
    try:
        torch.save(checkpoint, checkpoint_path)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected. Saving checkpoint...')
        torch.save(checkpoint, checkpoint_path)
        print('Checkpoint saved.')
        raise

if os.path.exists(checkpoint_path):
    print(f'Loading Checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    value_model_a.load_state_dict(checkpoint['value_model_a'])
    value_model_b.load_state_dict(checkpoint['value_model_b'])
    old_value_model_a.load_state_dict(checkpoint['old_value_model_a'])
    old_value_model_b.load_state_dict(checkpoint['old_value_model_b'])
    action0_model.load_state_dict(checkpoint['action0_model'])
    action1_model.load_state_dict(checkpoint['action1_model'])
    value_optimizer_a.load_state_dict(checkpoint['value_optimizer_a'])
    value_optimizer_b.load_state_dict(checkpoint['value_optimizer_b'])
    action0_optimizer.load_state_dict(checkpoint['action0_optimizer'])
    action1_optimizer.load_state_dict(checkpoint['action1_optimizer'])
    batch = checkpoint['batch']
    horizon = checkpoint['horizon']
else:
    save_checkpoint()

# for param_group in value_optimizer.param_groups:
#     param_group['lr'] = 1e-4

# horizon = 1
# batch = 0

batch_size = 5000
batch_count = 20
epoch_count = 1
minibatch_size = 1000
time_step = 0.1
minibatch_count = batch_size // minibatch_size
print('minibatch_count',minibatch_count)
cuda_generator = torch.Generator(device='cuda')
data_generator = DataGenerator(old_value_model_a,old_value_model_b,batch_size,time_step)
last_log_time = time.perf_counter()
quality_threshold = 0.985
value_quality_a = 0
value_quality_b = 0

print('Training...')
for _ in range(100000000):
    start_time = time.perf_counter()
    full_state,full_value_a,full_value_b,full_action0,full_action1 = data_generator.generate(horizon)
    dataset = TensorDataset(full_state,full_value_a,full_value_b,full_action0,full_action1)
    loader = DataLoader(dataset,batch_size=minibatch_size,shuffle=True,generator=cuda_generator)
    for epoch in range(epoch_count):
        for state, value_a, value_b, action0, action1 in loader:
            value_optimizer_a.zero_grad()
            value_optimizer_b.zero_grad()
            action0_optimizer.zero_grad()
            action1_optimizer.zero_grad()
            value_logit_a = value_model_a(state)
            value_logit_b = value_model_b(state)
            value_loss_a = F.binary_cross_entropy_with_logits(value_logit_a, value_a)
            value_loss_b = F.binary_cross_entropy_with_logits(value_logit_b, value_b)
            value_loss_a.backward()
            value_loss_b.backward()
            value_optimizer_a.step()
            value_optimizer_b.step()
            # Focus on training the value model for now.
            # if horizon > 0: 
            #     action0_logit = action0_model(state)
            #     action1_logit = action1_model(state)
            #     action0_loss = F.cross_entropy(action0_logit, action0)
            #     action1_loss = F.cross_entropy(action1_logit, action1)
            #     action_loss = action0_loss + action1_loss
            #     action_loss.backward()
            #     action0_optimizer.step()
            #     action1_optimizer.step()
        with torch.no_grad():
            full_value_estimate_a = torch.sigmoid(value_model_a(full_state))
            full_value_estimate_b = torch.sigmoid(value_model_b(full_state))
            full_value_mse_a = F.mse_loss(full_value_estimate_a, full_value_a)
            full_value_mse_b = F.mse_loss(full_value_estimate_b, full_value_b)
            null_value_estimate_a = full_value_a.mean()
            null_value_estimate_b = full_value_b.mean()
            null_value_mse_a = ((full_value_a - null_value_estimate_a)**2).mean()
            null_value_mse_b = ((full_value_b - null_value_estimate_b)**2).mean()
            value_quality_a = 1 - full_value_mse_a / null_value_mse_a
            value_quality_b = 1 - full_value_mse_b / null_value_mse_b
            full_action0_logit = action0_model(full_state)
            full_action1_logit = action1_model(full_state)
            full_choice0 = full_action0_logit.argmax(dim=1)
            full_choice1 = full_action1_logit.argmax(dim=1)
            action0_quality = (full_choice0 == full_action0).float().mean()
            action1_quality = (full_choice1 == full_action1).float().mean()
            message = ''
            message += f'Horizon: {horizon}, '
            message += f'Batch: {batch+1}, '
            message += f'ValueA: {value_quality_a:.03f}, '
            message += f'ValueB: {value_quality_b:.03f}, '
            now = time.perf_counter()
            message += f'Time: {now - last_log_time:.03f}, '
            last_log_time = now
            print(message)
    save_checkpoint()
    min_quality = min(value_quality_a, value_quality_b)
    if batch + 1 >= batch_count and min_quality > quality_threshold:
        print(f'Horizon {horizon} Complete.')
        horizon += 1
        batch = 0
        print(f'Saving checkpoint...')
        old_value_model_a.load_state_dict(value_model_a.state_dict())
        old_value_model_b.load_state_dict(value_model_b.state_dict())
        save_checkpoint()
        print(f'Beginning Horizon {horizon}...')
        continue
    batch += 1