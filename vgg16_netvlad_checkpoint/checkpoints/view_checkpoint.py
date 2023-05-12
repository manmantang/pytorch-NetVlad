import torch

model_dict  = torch.load('checkpoint.pth.tar',map_location = 'gpu')
for name, para in model_dict.items():
    print(name,':',para.size())
