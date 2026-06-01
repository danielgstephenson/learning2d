
import sys
from typing import Any
from math import log
from matplotlib.pylab import permutation
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.func import vmap, grad
import os
import time

from generator import DataGenerator
from models import ActionModel, ValueModel

sys.stdout = open('train.log', 'w', buffering=1)

checkpoint_path = './checkpoints/checkpoint.pt'
value_model = ValueModel()
target_value_model = ValueModel()
action0_model = ActionModel()
action1_model = ActionModel()
value_optimizer = torch.optim.AdamW(value_model.parameters(),lr=1e-4)
action0_optimizer = torch.optim.AdamW(action0_model.parameters(),lr=1e-4)
action1_optimizer = torch.optim.AdamW(action1_model.parameters(),lr=1e-4)
stage = 0
batch = 0

def save_checkpoint():
    checkpoint: dict[str, Any] = { 
        'value_model': value_model.state_dict(),
        'target_value_model': target_value_model.state_dict(),
        'action0_model': action0_model.state_dict(),
        'action1_model': action1_model.state_dict(),
        'value_optimizer': value_optimizer.state_dict(),
        'action0_optimizer': action0_optimizer.state_dict(),
        'action1_optimizer': action1_optimizer.state_dict(),
        'batch': batch,
        'stage': stage,
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
    value_model.load_state_dict(checkpoint['value_model'])
    target_value_model.load_state_dict(checkpoint['target_value_model'])
    action0_model.load_state_dict(checkpoint['action0_model'])
    action1_model.load_state_dict(checkpoint['action1_model'])
    value_optimizer.load_state_dict(checkpoint['value_optimizer'])
    action0_optimizer.load_state_dict(checkpoint['action0_optimizer'])
    action1_optimizer.load_state_dict(checkpoint['action1_optimizer'])
    batch = checkpoint['batch']
    stage = checkpoint['stage']
else:
    save_checkpoint()

# for param_group in value_optimizer.param_groups:
#     param_group['lr'] = 1e-4

# stage = 1
# batch = 0

batch_size = 5000
step_count = 300
batch_count = 10
epoch_count = 1
minibatch_size = 2000
time_step = 0.1
minibatch_count = (batch_size*step_count) // minibatch_size
print('minibatch_count',minibatch_count)
cuda_generator = torch.Generator(device='cuda')
data_generator = DataGenerator(
    target_value_model,action0_model,action1_model,
    batch_size,step_count,time_step
)
last_log_time = time.perf_counter()
quality_threshold = 0.5

entropy0 = []
entropy1 = []
value_quality = []

print('Training...')
for _ in range(100000000):
    start_time = time.perf_counter()
    data = data_generator.generate()
    for epoch in range(epoch_count):
        entropy0 = []
        entropy1 = []
        value_quality = []
        perm = torch.randperm(batch_size*step_count)
        starts =  range(0, batch_size*step_count, minibatch_size)
        for m, s in enumerate(starts):
            idx = perm[s:s+minibatch_size]
            state = data[0][idx]
            value = data[1][idx]
            action0 = data[2][idx]
            action1 = data[3][idx]
            advantage = data[4][idx]
            value_optimizer.zero_grad()
            value_logit = value_model(state)
            value_loss = F.binary_cross_entropy_with_logits(value_logit, value)
            value_loss.backward()
            value_optimizer.step()
            with torch.no_grad():
                value_estimate = torch.sigmoid(value_model(state))
                value_mse = F.mse_loss(value_estimate, value)
                null_value_estimate = value.mean()
                null_value_mse = ((value - null_value_estimate)**2).mean()
                r2 = 1 - value_mse / null_value_mse
                value_quality.append(r2.item())
            if stage > 0:
                action0_optimizer.zero_grad()
                action1_optimizer.zero_grad()
                logprob0 = action0_model.logprobs(state).gather(1,action0)
                logprob1 = action1_model.logprobs(state).gather(1,action1)
                action0_loss = -(logprob0*advantage).mean()
                action1_loss = +(logprob1*advantage).mean()
                action0_loss.backward()
                action1_loss.backward()
                torch.nn.utils.clip_grad_norm_(action0_model.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(action1_model.parameters(), max_norm=0.5)
                action0_optimizer.step()
                action1_optimizer.step()
                with torch.no_grad():
                    logprob0 = action0_model.logprobs(state)
                    logprob1 = action1_model.logprobs(state)
                    e0 = 1 + (logprob0.exp()*logprob0).sum(dim=1).mean()/log(9)
                    e1 = 1 + (logprob1.exp()*logprob1).sum(dim=1).mean()/log(9)
                    entropy0.append(e0.item())
                    entropy1.append(e1.item())
        message = ''
        message += f'Stage: {stage}, '
        message += f'Batch: {batch+1}, '
        message += f'Value: {np.mean(value_quality):.03f}, '
        if stage > 0:
            message += f'Action0: {np.mean(entropy0):.03f}, '
            message += f'Action1: {np.mean(entropy1):.03f}, '
        now = time.perf_counter()
        message += f'Time: {now - last_log_time:.03f}, '
        last_log_time = now
        print(message)
    save_checkpoint()
    if batch + 1 >= batch_count and np.mean(value_quality) > quality_threshold:
        print(f'Stage {stage} Complete.')
        stage += 1
        batch = 0
        target_value_model.load_state_dict(value_model.state_dict())
        save_checkpoint()
        print(f'Beginning Stage {stage}...')
        continue
    batch += 1