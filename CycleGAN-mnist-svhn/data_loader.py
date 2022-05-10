import torch
from torchvision import datasets
from torchvision import transforms

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform1 = transforms.Compose([
                    transforms.Resize(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform2 = transforms.Compose([
                    transforms.Resize(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])
    
    svhn  = datasets.SVHN (root=config.svhn_path,  download=True, transform=transform1)
    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform2)

    svhn_loader  = torch.utils.data.DataLoader(dataset=svhn,  batch_size=config.batch_size,
                                               shuffle=True,  num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=config.batch_size,
                                               shuffle=True,  num_workers=config.num_workers)
    
    return svhn_loader, mnist_loader