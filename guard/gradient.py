from typing import Any
import torch
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx
import torch.nn.functional as F
import torch.onnx
import os
from models import ValueModel, state_size

print('torch.__version__ =', torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = " + str(device))
torch.set_printoptions(sci_mode=False, precision=4)

value_model = ValueModel().cpu().eval()
value_model.requires_grad_(False)
checkpoint_path = './checkpoints/checkpoint.pt'
if os.path.exists(checkpoint_path):
    print('Loading Value Checkpoint...')
    value_checkpoint = torch.load(checkpoint_path, weights_only=False)
    value_model.load_state_dict(value_checkpoint['gen_model'])

def value_sum(state: Tensor) -> Tensor:
    return value_model(state).sum()

def compute_grad(state: Tensor) -> Tensor:
    return torch.func.grad(value_sum)(state)[:, 0:2]

dummy_input = torch.randn(1, state_size).cpu()
traced_graph = make_fx(compute_grad)(dummy_input)
traced_graph.eval()

base_path = 'onnx/guard.onnx'

print("Starting ONNX Export...")
try:
    batch_dim = torch.export.Dim("batch_size", min=1)
    onnx_program: Any = torch.onnx.export(
        traced_graph,
        (dummy_input,),
        dynamo=True,
        dynamic_shapes=({0: batch_dim},),
        input_names=['state'],
        output_names=['grad']
    )
    onnx_program.save(base_path)
    print(f'Saved: {os.path.getsize(base_path)/1e6:.3f} MB')
except Exception as e:
    import traceback
    traceback.print_exc()
