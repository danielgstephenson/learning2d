import torch
from torch import nn
from typing import Any


def save_value_checkpoint(
        path: str,
        model: nn.Module,
        old_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        horizon: float,
    ):
    checkpoint: dict[str, Any] = { 
        'model_state_dict': model.state_dict(),
        'old_model_state_dict': old_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'horizon': horizon,
    }
    try:
        torch.save(checkpoint, path)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected. Saving checkpoint...')
        torch.save(checkpoint, path)
        print('Checkpoint saved.')
        raise

def save_action_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
    checkpoint: dict[str, Any] = { 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    try:
        torch.save(checkpoint, path)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected. Saving checkpoint...')
        torch.save(checkpoint, path)
        print('Checkpoint saved.')
        raise