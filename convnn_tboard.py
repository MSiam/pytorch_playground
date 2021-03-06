import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from logger import Logger

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

logger= Logger('ptorch_summaries/')
class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.s1= nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.s2= nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1= nn.Linear(7*7*32, num_classes)

    def forward(self, inputs):
        x= self.s1(inputs)
        x= self.s2(x)
        x = x.reshape(x.size(0), -1)
        x= self.fc1(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model= ConvNeuralNet(num_classes).to(device)
criterion= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        outputs= model(images)
        loss=criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.scalar_summary('loss', loss, i)

        info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }
        for tag, images in info.items():
            logger.image_summary(tag, images, i)
        print('loss ', loss.item())

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs= model(images)
        _, predicted = torch.max(outputs, 1)
        correct+= (predicted==labels).sum()
        total+= labels.size(0)

    print('Accuracy ', float(correct)/total)

torch.save(model.state_dict(), 'log_reg.ckpt')
