from typing import Any
import torch
from torch import nn, Tensor
from torch.fx.experimental.proxy_tensor import make_fx
import torch.nn.functional as F
import torch.onnx
import sys

# log_file = open("export_debug.log", "w", encoding="utf-8")
# sys.stdout = log_file
# sys.stderr = log_file

print('torch.__version__ =', torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = " + str(device))
physics_dtype = torch.float32
torch.set_printoptions(sci_mode=False, precision=4)

class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 26
        k = 512
        self.projection = nn.Linear(self.input_dim, k)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(k) for _ in range(4)])
        self.hidden_layers = nn.ModuleList([nn.Linear(k, k) for _ in range(4)])
        self.output_layer = nn.Linear(k, 1)
    def forward(self, x: Tensor)->Tensor:
        x = self.projection(x)
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.silu(norm(x)))
        return self.output_layer(x)
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    
value_model = ValueModel().cpu().eval()
value_model.requires_grad_(False)

def compute_sum(x_in: Tensor) -> Tensor:
    return value_model(x_in).sum()

def compute_grad(x_in: Tensor) -> Tensor:
    return torch.func.grad(compute_sum)(x_in)

test_input = torch.randn(1, 26).cpu()
test_grad = compute_grad(test_input)
print('test_grad calculation successful.')

print("\nExtracting Raw ATen Graph via make_fx...")
dummy_input = torch.randn(1, 26).cpu()

# make_fx captures a functional FX graph of a Python function by tracing it with "proxy" tensors
traced_grad_graph = make_fx(compute_grad)(dummy_input)

print("Starting ONNX Export (The FX Proxy Path)...")
try:
    batch_dim = torch.export.Dim("batch_size", min=1)
    onnx_program: Any = torch.onnx.export(
        traced_grad_graph,      
        (dummy_input,),     
        dynamo=True,
        dynamic_shapes=({0: batch_dim},)
    )
    onnx_program.save("onnx/grad_model.onnx")
    print("Export successful: grad_model.onnx")
except Exception as e:
    import traceback
    traceback.print_exc()