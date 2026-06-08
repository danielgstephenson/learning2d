
import sys
from typing import Any
from math import log
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import vmap, grad
import os
import time

from generator import DataGenerator
from models import ValueModel

sys.stdout = open('train.log', 'w', buffering=1)

checkpoint_path = './checkpoints/checkpoint.pt'
model0 = ValueModel()
model1 = ValueModel()
batch = 0
stage = 0

batch_size = 25000
epoch_count = 1
minibatch_size = 2000
target_discount = 1/1000
target_noise = 1/100
quality_threshold = 0.98

gen = DataGenerator(batch_size)
gen.discount = 1/2
gen.model0.noise = 1.0
gen.model1.noise = 1.0
opt0 = torch.optim.AdamW(gen.model0.parameters(),lr=1e-4)
opt1 = torch.optim.AdamW(gen.model1.parameters(),lr=1e-4)
step_count = gen.step_count
minibatch_count = (batch_size*gen.step_count) // minibatch_size
print('minibatch_count',minibatch_count)
cuda_generator = torch.Generator(device='cuda')

def save_checkpoint():
    checkpoint: dict[str, Any] = {
        'model0': model0.state_dict(),
        'model1': model1.state_dict(),
        'gen_model0': gen.model0.state_dict(),
        'gen_model1': gen.model1.state_dict(),
        'discount': gen.discount,
        'noise0': gen.model0.noise,
        'noise1': gen.model1.noise,
        'opt0': opt0.state_dict(),
        'opt1': opt1.state_dict(),
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
    model0.load_state_dict(checkpoint['model0'])
    model1.load_state_dict(checkpoint['model1'])
    gen.model0.load_state_dict(checkpoint['gen_model0'])
    gen.model1.load_state_dict(checkpoint['gen_model1'])
    gen.discount = checkpoint['discount']
    gen.model0.noise = checkpoint['noise0']
    gen.model1.noise = checkpoint['noise1']
    opt0.load_state_dict(checkpoint['opt0'])
    opt1.load_state_dict(checkpoint['opt1'])
    batch = checkpoint['batch']
    stage = checkpoint['stage']
else:
    save_checkpoint()

def reset(phase: int):
    model = gen.model0 if phase==0 else gen.model1
    opt = opt0 if phase==0 else opt1
    model.load_state_dict(ValueModel().state_dict())
    model.noise = 1.0
    gen.discount = 1/2
    fresh_opt = torch.optim.AdamW(model.parameters(),lr=1e-4)
    opt.load_state_dict(fresh_opt.state_dict())

# stage = 0
# batch = 0
# discount = 1/2
# noise = 1
# value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=1e-4)

targets = []
estimates = []
qualities = []

last_log_time = time.perf_counter()
print('Training...')
for _ in range(100000000):
    start_time = time.perf_counter()
    data = gen.generate()
    phase = stage % 2
    model = gen.model0 if phase==0 else gen.model1    
    opt = opt0 if phase==0 else opt1
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
            logit = model(state)
            loss = F.binary_cross_entropy_with_logits(logit, value)
            loss.backward()
            opt.step()
            with torch.no_grad():
                estimate = torch.sigmoid(model(state))
                mse = F.mse_loss(estimate, value)
                null_estimate = value.mean()
                null_mse = ((value - null_estimate)**2).mean()
                r2 = 1 - mse/null_mse
                targets.append(value.mean().item())
                estimates.append(estimate.mean().item())
                qualities.append(r2.item())
        charge = torch.mean(gen.world.charge).item()
        message = ''
        message += f'stage: {stage}, '
        message += f'batch: {batch+1}, '
        message += f'R2: {np.mean(qualities):.03f}, '
        message += f'p: {gen.discount:.05f}, '
        message += f'noise: {model.noise:.05f}, '
        now = time.perf_counter()
        message += f'Time: {now - last_log_time:.03f}, '
        last_log_time = now
        print(message)
    save_checkpoint()
    meanQuality = np.mean(qualities)
    gen.discount = max(0.99*gen.discount, target_discount)
    model.noise = max(0.99*model.noise, target_noise)
    ready = meanQuality > quality_threshold
    ready = ready and (gen.discount == target_discount)
    ready = ready and (model.noise == target_noise)
    if ready:
        print(f'Stage {stage} Complete.')
        learner = model0 if phase==0 else model1 
        learner.load_state_dict(model.state_dict())
        stage += 1
        phase = stage % 2
        batch = 0
        reset(phase)
        save_checkpoint()
        print(f'Beginning Stage {stage}...')
        continue
    batch += 1