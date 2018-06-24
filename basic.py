import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

#x= torch.tensor(1, requires_grad=True)
#w= torch.tensor(2, requires_grad=True)
#b= torch.tensor(3, requires_grad=True)
#
#y= w*x+b
#y.backward()
#
#print(x.grad)
#print(w.grad)
#print(b.grad)
###################################

#x = torch.randn(10,3)
#y = torch.randn(10,2)
#
#linear= nn.Linear(3,2)
#print(linear.weight)
#print(linear.bias)
#
#criterion= nn.MSELoss()
#optimizer= torch.optim.SGD(linear.parameters(), lr= 0.01)
#
#pred= linear(x)
#loss= criterion(pred, y)
#print('loss', loss)
#
#loss.backward()
#print('dl/dw ', linear.weight.grad)
#print('dl/db ', linear.bias.grad)
#
#optimizer.step()
##linear.weight.data.sub_(0.01*linear.weight.grad.data)
##linear.bias.data.sub_(0.01*linear.bias.grad.data)
#
#pred= linear(x)
#loss= criterion(pred, y)
#print('loss', loss)
#############################################33

x= np.array([[1,2], [3,4]])
y=torch.from_numpy(x)
z= y.numpy()

print(y)
print(z)
