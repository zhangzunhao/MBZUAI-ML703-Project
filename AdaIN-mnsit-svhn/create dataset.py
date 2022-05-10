import torchvision
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import cv2
class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)
# transform_mnist=transforms.Compose([
#         GrayscaleToRgb(),
#         transforms.ToTensor(),
#         transforms.Resize(64),
#         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# mnist_dataset = torchvision.datasets.MNIST(root='mnis', train=True, download=True, transform=transform_mnist)
# for iteration in range(60000):
#     pic, _ = mnist_dataset[iteration]
#     torchvision.utils.save_image(pic, './mnist/mnist%d.jpeg' % (iteration + 1),  normalize=True)
    
# transform_svhn = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize(64),
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# svhn_dataset  = torchvision.datasets.SVHN(root=None, download=True, transform=transform_svhn)
# for iteration in range(60000):
#     pic, _ = svhn_dataset[iteration]
#     torchvision.utils.save_image(pic, './svhn/svhn%d.jpeg' % (iteration + 1),  normalize=True)
img = cv2.imread('./output/mnist1.jpg')
trans1 = torchvision.transforms.ToTensor()
img = trans1(img)
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
imshow(img)