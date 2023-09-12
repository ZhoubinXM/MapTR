import torch
from mmcv.cnn.utils.flops_counter import * 

file_path = './ckpts/maptr_tiny_r50_110e.pth'
model = torch.load(file_path, map_location='cpu')
all = 0
for key in list(model['state_dict'].keys()):
    all += model['state_dict'][key].nelement()
print(all)

model_param = params_to_string(all)
print(model_param)

# smaller 63374123
# v4 69140395
