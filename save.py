import torch
from torch import nn


def save_checkpoint(
        model: nn.Module,
        old_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        discount: float, 
        horizon: float, 
        path: str
    ):
    checkpoint = { 
        'model_state_dict': model.state_dict(),
        'old_model_state_dict': old_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'discount': discount,
        'horizon': horizon
    }
    try:
        torch.save(checkpoint, path)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected. Saving checkpoint...')
        torch.save(checkpoint, path)
        print('Checkpoint saved.')
        raise