"""
Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
"""
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, SVHN
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from tqdm import tqdm

import config
from models import Net
from utils import GrayscaleToRgb, GradientReversal
import cv2
import torchvision.transforms as transforms


device = torch.device('cuda')


def main(args):
    model = Net().to(device)
    model.load_state_dict(torch.load('./trained_models/source.pt'))
    #model.load_state_dict(torch.load('./trained_models/revgrad_svhn.pt'))
    
    feature_extractor = model.feature_extractor
    classifier = model.classifier

    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(256, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    
    # transform1 = transforms.Compose([
    #     #transforms.Resize([32,32]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(0.5,0.5)])

    # train_length=60000
    # xtrain=torch.empty(train_length,3,32,32)
    # for i in range(train_length):
    #     img = cv2.imread('c:/Users/ZHANG/Desktop/pytorch-AdaIN/output/mnist%d.jpg' % (i + 1))  
    #     img = cv2.resize(img, (32,32))
    #     xtrain[i]=transform1(img)

    # ytrain=torch.empty(train_length)
    # mnist_dataset = MNIST(root='./mnist', train=True, download=True)
    # for i in range(train_length):
    #     _, label = mnist_dataset[i]
    #     ytrain[i] = label
    # source_dataset = torch.utils.data.TensorDataset(xtrain.float(), ytrain.long())
    
    source_dataset = MNIST(root='./mnist', train=True, download=True,
                          transform=Compose([Resize(32), GrayscaleToRgb(), ToTensor()]))
    source_loader = DataLoader(source_dataset, batch_size = half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    
    #target_dataset = MNISTM(train=False)
    target_dataset = SVHN(root='./svhn', download=True, transform=Compose([Resize(32), ToTensor()]))
    target_loader = DataLoader(target_dataset, batch_size = half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)    
    

    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))

    for epoch in range(1, args.epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))        

        total_domain_loss = total_label_accuracy = 0
        for (source_inputs, source_labels), (target_inputs, _) in tqdm(batches, leave=False, total=n_batches):
            
            x = torch.cat([source_inputs, target_inputs]).to(device)
            domain_y = torch.cat([torch.ones(source_inputs.shape[0]), torch.zeros(target_inputs.shape[0])]).to(device)
            source_labels = source_labels.to(device)
            
            features = feature_extractor(x).view(x.shape[0], -1)
            domain_pred = discriminator(features).squeeze()
            label_pred = classifier(features[:source_inputs.shape[0]])
            
            domain_loss = F.binary_cross_entropy_with_logits(domain_pred, domain_y)
            label_loss = F.cross_entropy(label_pred, source_labels)
            loss = domain_loss + label_loss
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_domain_loss += domain_loss.item()
            total_label_accuracy += (label_pred.max(1)[1] == source_labels).float().mean().item()

        domain_loss = total_domain_loss / n_batches
        source_accuracy = total_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch}: domain_loss={domain_loss:.4f},source_accu={source_accuracy:.4f}')

        #torch.save(model.state_dict(), 'trained_models/revgrad_adain_svhn.pt')
        torch.save(model.state_dict(), 'trained_models/revgrad_svhn.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')    
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=15)
    args = arg_parser.parse_args()
    main(args)
