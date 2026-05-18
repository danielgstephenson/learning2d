import torch
from torch import nn
from typing import Any

def save_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: int,
        horizon: int,
    ):
    checkpoint: dict[str, Any] = { 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch': batch,
        'horizon': horizon,
    }
    try:
        torch.save(checkpoint, path)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected. Saving checkpoint...')
        torch.save(checkpoint, path)
        print('Checkpoint saved.')
        raise