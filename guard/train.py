
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
from models import ValueModel

sys.stdout = open('train.log', 'w', buffering=1)

checkpoint_path = './checkpoints/checkpoint.pt'
value_model = ValueModel()
target_value_model = ValueModel()
value_optimizer = torch.optim.AdamW(value_model.parameters(),lr=1e-4)
stage = 0
batch = 0

def save_checkpoint():
    checkpoint: dict[str, Any] = { 
        'value_model': value_model.state_dict(),
        'target_value_model': target_value_model.state_dict(),
        'value_optimizer': value_optimizer.state_dict(),
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
    value_optimizer.load_state_dict(checkpoint['value_optimizer'])
    batch = checkpoint['batch']
    stage = checkpoint['stage']
else:
    save_checkpoint()

# for param_group in value_optimizer.param_groups:
#     param_group['lr'] = 1e-4

stage = 1
batch = 0

batch_size = 5000
discount_rate = 1/300
state_noise = 1
action_noise = 0.1
step_count = 10
batch_count = 10
epoch_count = 1
minibatch_size = 2000
time_step = 0.1
minibatch_count = (batch_size*step_count) // minibatch_size
print('minibatch_count',minibatch_count)
cuda_generator = torch.Generator(device='cuda')
data_generator = DataGenerator(
    target_value_model,batch_size,step_count,
    discount_rate,state_noise,action_noise,time_step
)
last_log_time = time.perf_counter()
quality_threshold = 0.8

targets = []
estimates = []
qualities = []

print('Training...')
for _ in range(100000000):
    start_time = time.perf_counter()
    data = data_generator.generate(stage)
    for epoch in range(epoch_count):
        targets = []
        estimates = []
        qualities = []
        perm = torch.randperm(batch_size*step_count)
        starts =  range(0, batch_size*step_count, minibatch_size)
        for m, s in enumerate(starts):
            idx = perm[s:s+minibatch_size]
            state = data[0][idx]
            value = data[1][idx]
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
                targets.append(value.mean().item())
                estimates.append(value_estimate.mean().item())
                qualities.append(r2.item())
        message = ''
        message += f'stage: {stage}, '
        message += f'batch: {batch+1}, '
        message += f'R2: {np.mean(qualities):.03f}, '
        now = time.perf_counter()
        message += f'Time: {now - last_log_time:.03f}, '
        last_log_time = now
        print(message)
    save_checkpoint()
    if batch + 1 >= batch_count and np.mean(qualities) > quality_threshold:
        print(f'Stage {stage} Complete.')
        stage += 1
        batch = 0
        target_value_model.load_state_dict(value_model.state_dict())
        save_checkpoint()
        print(f'Beginning Stage {stage}...')
        continue
    batch += 1