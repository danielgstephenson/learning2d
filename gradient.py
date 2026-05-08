from typing import Any
import torch
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx
import torch.nn.functional as F
import torch.onnx
import os
from value import ValueModel

# log_file = open("export_debug.log", "w", encoding="utf-8")
# sys.stdout = log_file
# sys.stderr = log_file

print('torch.__version__ =', torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = " + str(device))
physics_dtype = torch.float32
torch.set_printoptions(sci_mode=False, precision=4)
    
value_model = ValueModel().cpu().eval()
value_model.requires_grad_(False)
value_checkpoint_path = './checkpoints/value_checkpoint.pt'
if os.path.exists(value_checkpoint_path):
    print('Loading Value Checkpoint...')
    value_checkpoint = torch.load(value_checkpoint_path, weights_only=False)
    value_model.load_state_dict(value_checkpoint['model_state_dict'])

def value_sum(state: Tensor)->Tensor:
    return value_model(state).sum()

def compute_grad(state: Tensor) -> Tensor:
    return torch.func.grad(value_sum)(state)[:,8:10]

test_input = torch.randn(1, 26).cpu()
test_grad = compute_grad(test_input)

print('test_input:',test_input)
print('test_grad:',test_grad)

dummy_input = torch.randn(1, 26).cpu()
traced_graph = make_fx(compute_grad)(dummy_input)

print("Starting ONNX Export...")
try:
    batch_dim = torch.export.Dim("batch_size", min=1)
    onnx_program: Any = torch.onnx.export(
        traced_graph,      
        (dummy_input,),     
        dynamo=True,
        dynamic_shapes=({0: batch_dim},)
    )
    onnx_program.save("onnx/grad_model.onnx")
    print("Export successful: grad_model.onnx")
except Exception as e:
    import traceback
    traceback.print_exc()