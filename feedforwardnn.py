import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Hyper-parameters
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
hidden_size= 500

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1= nn.Linear(input_size, hidden_size)
        self.relu1= nn.ReLU()
        self.fc2= nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        x= self.fc1(inputs)
        x= self.relu1(x)
        x= self.fc2(x)
        return x

model= NeuralNet(input_size, hidden_size, num_classes)
criterion= nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images= images.reshape((-1,28*28))
        outputs= model(images)
        loss=criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('loss ', loss.item())

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images= images.reshape(-1,28*28)
        outputs= model(images)
        _, predicted = torch.max(outputs, 1)
        correct+= (predicted==labels).sum()
        total+= labels.size(0)

    print('Accuracy ', float(correct)/total)

torch.save(model.state_dict(), 'log_reg.ckpt')
