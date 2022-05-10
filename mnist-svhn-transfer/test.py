import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from model import G21,ResNet,BasicBlock

def test():
    g21 = G21(conv_dim=64).cuda()
    PATH = './g21-25000-reconst.pkl'; g21.load_state_dict(torch.load(PATH)); g21.eval()
    
    model = ResNet(BasicBlock, [2, 2, 2, 2]).cuda()   
    PATH = './mnist_resnet18.pth'; model.load_state_dict(torch.load(PATH)); model.eval()
    
    transform1 = transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    svhn  = datasets.SVHN ('./svhn',  download=True, transform=transform1)
    svhn_loader = torch.utils.data.DataLoader(dataset=svhn, batch_size=64, shuffle=True, num_workers=2)
    
    correct = 0
    total = 0
    
    with torch.no_grad():   
        for i, data in enumerate(svhn_loader):        
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()            
            inputs = g21(inputs)
            resize = transforms.Resize(28)
            outputs = resize(inputs)
            outputs = model(inputs)
            
            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            
            correct += (predicted == labels).sum().item()
    total_num_img = i * 64
    print('accuracy:',correct / total_num_img)
    
if __name__ == '__main__':
    test()
    
    