import torch

def getDevice() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    
    else:
        return torch.device("cpu")

