# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_data.ipynb.

# %% auto 0
__all__ = ['load_mnist']

# %% ../nbs/00_data.ipynb 5
from torchvision import datasets
from torchvision.transforms import ToTensor

def load_mnist():
    mnist_train = datasets.MNIST(root="datasets", train=True, 
                                download=True, transform=ToTensor())
    mnist_val = datasets.MNIST(root="datasets", train=False,
                            transform=ToTensor())
    return mnist_train, mnist_val
