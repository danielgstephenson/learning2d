from typing import Any
import torch
from torch import Tensor, nn
from torch.fx.experimental.proxy_tensor import make_fx
import os
from value import ValueModel, state_size

torch.set_default_device('cpu')
model = ValueModel().cpu().eval()

checkpoint_path = './checkpoints/attack.pt'
if os.path.exists(checkpoint_path):
    print(f'Loading Checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])

def action_values(state: Tensor)->Tensor:
    return model.action_values(state)

example = torch.randn(2, state_size)          # batch of 2, not 1
batch_dim = torch.export.Dim('batch_size', min=1)
graph = make_fx(action_values, tracing_mode='symbolic', _allow_non_fake_inputs=True)(example)
graph.eval()

onnx_program = torch.onnx.export(
    graph, (example,), dynamo=True,
    dynamic_shapes=({0: batch_dim},),
    input_names=['state'], output_names=['action_values'],
)
if onnx_program is None:
    raise RuntimeError('torch.onnx.export returned None (dynamo export did not run)')

onnx_path = './onnx/attack.onnx'
onnx_program.save(onnx_path)
print(f'Saved: {os.path.getsize(onnx_path)/1e6:.3f} MB')

import onnx
dims = [d.dim_param or d.dim_value
        for d in onnx.load(onnx_path).graph.input[0].type.tensor_type.shape.dim]
print('state:', dims)  