import torch
import numpy as np
import onnxruntime as ort
from torch.func import vmap, grad as fgrad
from models import ValueModel, state_size
import os

# Load PyTorch model
value_model = ValueModel().cpu().eval()
value_model.requires_grad_(False)
checkpoint = torch.load('./checkpoints/checkpoint.pt', weights_only=False)
value_model.load_state_dict(checkpoint['value_model'])

# State from JS console log
state_np = np.array([[
0.06183893920030503, 
-0.33127153033611006, 
0.000023930345492259594, 
-1.0882646771279569e-7, 
0.07514239488294373, 
0.021465080559556204, 
-0.2829090934341776, 
1.8184974404324812, 
127.86074699447897, 
-65.1988703941824, 
-29.131775203066155, 
11.396992365643193, 
121.99303309053533, 
-61.642229231115095, 
-24.684079651566577, 
16.785309219010934, 
1, 
1, 
116.68553098489969, 
11.396992371132974, 
91.11619816893005, 
131.64496695030243, 
-29.13177641023944, 
211.39699237113297, 
-170.55313264754892, 
152.81834860844248, 
-229.13177641023944, 
11.396992371132995, 
-170.55313264754898, 
-130.02436386617651, 
-29.131776410239468, 
-114.3578178378323, 
69.10617222409371, 
-86.84095626320024
]],
dtype=np.float32)

state_t = torch.tensor(state_np)

# PyTorch gradient
gradient_fn = vmap(fgrad(lambda x: value_model(x).sum()))
g_pytorch = gradient_fn(state_t)[0, 0:2].detach().numpy()
print(f'PyTorch gradient: {g_pytorch}')

# ONNX (float32, unquantized) gradient
if os.path.exists('onnx/grad_model.onnx'):
    session = ort.InferenceSession('onnx/grad_model.onnx')
    g_onnx = session.run(['grad'], {'state': state_np})[0]
    print(f'ONNX float32 gradient: {g_onnx}')

# INT8 gzip model — decompress first
import gzip, shutil
if os.path.exists('onnx/guard.onnx.gz'):
    with gzip.open('onnx/guard.onnx.gz', 'rb') as f_in:
        with open('onnx/guard_tmp.onnx', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    session_int8 = ort.InferenceSession('onnx/guard_tmp.onnx')
    g_int8 = session_int8.run(['grad'], {'state': state_np})[0]
    print(f'ONNX INT8 gradient:    {g_int8}')
    os.remove('onnx/guard_tmp.onnx')
