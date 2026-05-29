from typing import Any
import torch
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx
import torch.nn.functional as F
import torch.onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
from models import ValueModel, state_size

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
checkpoint_path = './checkpoints/checkpoint.pt'
if os.path.exists(checkpoint_path):
    print('Loading Value Checkpoint...')
    value_checkpoint = torch.load(checkpoint_path, weights_only=False)
    value_model.load_state_dict(value_checkpoint['model_state_dict'])

def value_sum(state: Tensor)->Tensor:
    return value_model(state).sum()

def compute_grad(state: Tensor) -> Tensor:
    return torch.func.grad(value_sum)(state)[:,0:2]

test_input = torch.randn(1, state_size).cpu()
test_grad = compute_grad(test_input)
print('test_input:',test_input)
print('test_grad:',test_grad)

dummy_input = torch.randn(1, state_size).cpu()
traced_graph = make_fx(compute_grad)(dummy_input)

base_path = 'onnx/grad_model.onnx'
quant_path = 'onnx/grad_model_quant.onnx'

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
    print(f"Export successful: {base_path}")
    onnx_program.save(base_path)
    quantize_dynamic(
        model_input=base_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantization successful: {quant_path}")
except Exception as e:
    import traceback
    traceback.print_exc()