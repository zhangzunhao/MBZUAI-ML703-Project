import torchvision
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
from resnet import ResNet, BasicBlock

transform1 = transforms.Compose([
    #transforms.Resize([32,32]),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)])

train_length=20000
xtrain=torch.empty(train_length,3,32,32)
for i in range(train_length):
    img = cv2.imread('./output/mnist%d.jpg' % (i + 1))  
    img = cv2.resize(img, (32,32))
    xtrain[i]=transform1(img)

ytrain=torch.empty(train_length)
mnist_dataset = torchvision.datasets.MNIST(root='mnis', train=True, download=True)
for i in range(train_length):
    _, label = mnist_dataset[i]
    ytrain[i] = label
trainset = torch.utils.data.TensorDataset(xtrain.float(), ytrain.long())
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=4, shuffle=False)

transform2 = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)])

svhn_set = torchvision.datasets.SVHN(root='svhn', download=True, transform=transform2)

from sklearn.model_selection import train_test_split
_,testset = train_test_split(svhn_set ,train_size = 0.8)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)


model = ResNet(BasicBlock, [2, 2, 2, 2]).cuda() 
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()        
        
        optimizer.zero_grad() # Set the gradients of all optimized torch.Tensor s to zero.

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

#PATH = './augment_net.pth'; torch.save(model.state_dict(), PATH)
    
correct = 0
total = 0

with torch.no_grad():   # since we're not training, we don't need to calculate the gradients for our outputs
    for i, data in enumerate(testloader):        
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        
        # The class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        
        correct += (predicted == labels).sum().item()
total_num_img = i * 4
print('accuracy:',correct / total_num_img)