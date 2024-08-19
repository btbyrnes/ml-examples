import torch, torchvision
from pathlib import Path

DATA_PATH = Path(".","data")

def download():
    # torchvision.datasets.mnist.MNIST()
    path = Path(".", "data")

    if not path.exists():
        path.mkdir()

    torchvision.datasets.MNIST(root=path, download=True)
 

def dataset(transform=None) -> torch.utils.data.Dataset:
    return torchvision.datasets.MNIST(DATA_PATH, transform=transform)



if __name__ == "__main__":
    pass