from typing import Any
import torch
from torch import Tensor
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

dummy_input = torch.randn(1, state_size).cpu()
traced_graph = make_fx(action_values)(dummy_input)
traced_graph.eval()

onnx_path = './onnx/attack.onnx'

print("Starting ONNX Export...")
try:
    batch_dim = torch.export.Dim("batch_size", min=1)
    onnx_program: Any = torch.onnx.export(
        traced_graph,
        (dummy_input,),
        dynamo=True,
        dynamic_shapes=({0: batch_dim},),
        input_names=['state'],
        output_names=['action_values']
    )
    onnx_program.save(onnx_path)
    print(f'Saved: {os.path.getsize(onnx_path)/1e6:.3f} MB')
except Exception as e:
    import traceback
    traceback.print_exc()