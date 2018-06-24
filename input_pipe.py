import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import os
import cv2

#train_dataset= torchvision.datasets.CIFAR10(root='../../data/',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#image, label= train_dataset[0]
#print(image.size())
#print(label)
#train_loader= torch.utils.data.DataLoader(dataset= train_dataset, batch_size=64, shuffle=True)

######################################################3
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        main_dir= '/home/menna/Datasets/FORD-M/objects/daylight/translation/bottle1/'
        self.files= []
        for f in sorted(os.listdir(main_dir)):
            self.files.append((main_dir+f, 0))

    def __getitem__(self, index):
#        import pdb; pdb.set_trace()
        img= cv2.imread(self.files[index][0])
        img= cv2.resize(img, (640,480))
        img= img.transpose(2,0,1)
        label= self.files[index][1]
        return img, label

    def __len__(self):
        return len(self.files)

custom_dataset= CustomDataset()
train_loader= torch.utils.data.DataLoader(dataset= custom_dataset, batch_size= 10, shuffle=True)
#########################################################

data_iter= iter(train_loader)
images, labels= data_iter.next()

for images, labels in train_loader:
    print(images.size())
    print(labels)


