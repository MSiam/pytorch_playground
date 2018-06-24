import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

resnet= torchvision.models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad= False

resnet.fc= nn.Linear(resnet.fc.in_features, 100)

images= torch.randn(64,3,224,224)
outputs= resnet(images)
print(outputs.size())

###################################
#torch.save(resnet, 'model.ckpt')
#model = torch.load('model.ckpt')
#print('loaded ', model)

torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
