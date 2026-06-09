
import sys
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import os
import time

from generator import DataGenerator
from models import ValueModel

sys.stdout = open('train.log', 'w', buffering=1)

checkpoint_path = './checkpoints/checkpoint.pt'
model = ValueModel()
batch = 0
stage = 0

batch_size = 25000
batch_count = 10
epoch_count = 1
minibatch_size = 2000
target_discount = 1/1000
target_noise = 1/100
quality_threshold = 0.95

gen = DataGenerator(batch_size)
gen.discount = 1/2
gen.model.noise = 1.0
opt = torch.optim.AdamW(model.parameters(),lr=1e-4)
step_count = gen.step_count
minibatch_count = (batch_size*gen.step_count) // minibatch_size
print('minibatch_count',minibatch_count)
cuda_generator = torch.Generator(device='cuda')

def save_checkpoint():
    checkpoint: dict[str, Any] = {
        'model': model.state_dict(),
        'gen_model': gen.model.state_dict(),
        'discount': gen.discount,
        'noise': gen.model.noise,
        'opt': opt.state_dict(),
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
    model.load_state_dict(checkpoint['model'])
    gen.model.load_state_dict(checkpoint['gen_model'])
    gen.discount = checkpoint['discount']
    gen.model.noise = checkpoint['noise']
    opt.load_state_dict(checkpoint['opt'])
    batch = checkpoint['batch']
    stage = checkpoint['stage']
else:
    save_checkpoint()

targets = []
estimates = []
qualities = []

last_log_time = time.perf_counter()
print('Training...')
for _ in range(100000000):
    start_time = time.perf_counter()
    data = gen.generate(stage)
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
            opt.zero_grad()
            estimate = model(state)
            loss = F.mse_loss(estimate, value)
            loss.backward()
            opt.step()
            with torch.no_grad():
                estimate = model(state)
                mse = F.mse_loss(estimate, value)
                null_estimate = value.mean()
                null_mse = ((value - null_estimate)**2).mean()
                r2 = 1 - mse/null_mse
                targets.append(value.mean().item())
                estimates.append(estimate.mean().item())
                qualities.append(r2.item())
        message = ''
        message += f'stage: {stage}, '
        message += f'batch: {batch+1}, '
        message += f'R2: {np.mean(qualities):.03f}, '
        message += f'p: {gen.discount:.05f}, '
        message += f'noise: {gen.model.noise:.05f}, '
        now = time.perf_counter()
        message += f'Time: {now - last_log_time:.03f}, '
        last_log_time = now
        print(message)
    save_checkpoint()
    meanQuality = np.mean(qualities)
    if meanQuality > quality_threshold and batch >= batch_count:
        print(f'Stage {stage} Complete.')
        gen.model.load_state_dict(model.state_dict())
        stage += 1
        batch = 0
        gen.discount = 1/(stage+1)
        gen.model.noise = 0.1
        save_checkpoint()
        print(f'Beginning Stage {stage}...')
        continue
    batch += 1