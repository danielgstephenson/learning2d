
import sys
from typing import Any
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import os
import time

from generator import DataGenerator
from value import ValueModel

sys.stdout = open('train.log', 'w', buffering=1)

checkpoint_path = './checkpoints/checkpoint.pt'
model = ValueModel()
epoch = 0

stage_size = 100
epoch_size = 100_000
batch_size = 1000
target_discount = 1/4000
quality_threshold = 0.95

gen = DataGenerator(epoch_size)
gen.horizon = 0
gen.model.noise = 0.1
gen.step_count = 10
opt = torch.optim.AdamW(model.parameters(),lr=1e-3)
cuda_generator = torch.Generator(device='cuda')

def save_checkpoint():
    checkpoint: dict[str, Any] = {
        'model': model.state_dict(),
        'gen_model': gen.model.state_dict(),
        'horizon': gen.horizon,
        'noise': gen.model.noise,
        'opt': opt.state_dict(),
        'epoch': epoch,
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
    gen.horizon = checkpoint['horizon']
    gen.model.noise = checkpoint['noise']
    opt.load_state_dict(checkpoint['opt'])
    epoch = checkpoint['epoch']
else:
    save_checkpoint()

for g in opt.param_groups: 
    g['lr'] = 1e-3
gen.horizon = 1

last_log_time = time.perf_counter()
print('Training...')
for _ in range(100000000):
    state_data, value_data = gen.generate(gen.horizon)
    dataset = TensorDataset(state_data, value_data)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, generator=cuda_generator)
    with torch.no_grad():
        estimate = model(state_data)
        mse = torch.mean((estimate-value_data)**2)
        null_estimate = value_data.mean()
        null_mse = torch.mean((null_estimate-value_data)**2)
        r2 = (1 - mse/null_mse).item()
    for batch in dataloader:
        data: tuple[Tensor,Tensor] = batch
        state, value = data
        opt.zero_grad()
        estimate = model(state)
        mse = torch.mean((estimate-value)**2)
        mse.backward()
        opt.step()
    message = ''
    message += f'horizon: {gen.horizon:.01f}, '
    message += f'epoch: {epoch+1}, '
    message += f'R2: {r2:.03f}, '
    print(message)
    epoch += 1
    save_checkpoint()
    if epoch < stage_size: continue
    epoch = 0
    gen.model.load_state_dict(model.state_dict())
    gen.horizon = min(5, gen.horizon + 0.1)
    save_checkpoint()