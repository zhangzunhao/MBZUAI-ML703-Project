import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, SVHN
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

from data import MNISTM
from models import Net
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from utils import GrayscaleToRgb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):    
    "Choose dataset"
    #dataset = MNISTM(train=False)
    #dataset = datasets.MNIST(root='./mnist', train=False, download=True, transform=Compose([GrayscaleToRgb(), ToTensor()]))
    dataset = SVHN(root='./svhn', download=True, transform=Compose([Resize(32), ToTensor()]))    
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=1, pin_memory=True)
    
    "Choose the model"
    model = Net().to(device)
    #model.load_state_dict(torch.load('./trained_models/source.pt'))
    #model.load_state_dict(torch.load('./trained_models/source_mnistm.pt'))
    #model.load_state_dict(torch.load('./trained_models/revgrad_mnistm.pt'))
    model.load_state_dict(torch.load('./trained_models/revgrad_svhn.pt'))
    #model.load_state_dict(torch.load('./trained_models/revgrad_adain_svhn.pt'))
    model.eval()

    total_accuracy = 0
    with torch.no_grad():
        for x, y_true in dataloader:
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    
    mean_accuracy = total_accuracy / len(dataloader)
    print(f'Accuracy on target data: {mean_accuracy:.4f}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Test a model on MNIST-M')
    arg_parser.add_argument('--batch-size', type=int, default=256)
    args = arg_parser.parse_args()
    main(args)
