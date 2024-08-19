from typing import Callable
import torch, torch.optim as optim, torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data.dataloader as DataLoader
from model.vae import AutoEncoder

from torch.utils.tensorboard import SummaryWriter


def train(model:AutoEncoder, dl:DataLoader, loss_function:Callable, epochs:int=20, lr:float=0.01, device:torch.device=torch.device("cpu")) -> None:
    input_dim = next(model.encoder.parameters()).size()[1]

    writer = SummaryWriter()
    last_lr = lr

    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=0)
    for epoch in range(epochs):
        epoch_loss = 0
        kl_loss = 0
        re_loss = 0
        n = 0
        for batch, (x, _) in enumerate(dl):
            n += len(x)
            x = x.view(len(x), input_dim)
            x.to(device)

            opt.zero_grad()
            x_hat, mu, logVar = model(x)
            batch_re_loss, batch_kl_loss, loss = loss_function(x, x_hat, mu, logVar)

            re_loss += batch_re_loss
            kl_loss += batch_kl_loss

            epoch_loss += loss.item()
            loss.backward()
            opt.step()

        epoch_loss = epoch_loss / n
        kl_loss = kl_loss / n
        re_loss = re_loss / n
        this_lr = sched.get_last_lr()
        sched.step(epoch_loss)
        
        writer.add_scalar("Train/KL-loss", kl_loss , epoch)
        writer.add_scalar("Train/MSE-loss", re_loss , epoch)
        writer.add_scalar("Train/Total-loss", epoch_loss , epoch)
        writer.add_scalar("Train/LR", this_lr[0] , epoch)

    print(f"Epoch: {epoch:4d} - Loss: {epoch_loss:.6f}")


if __name__ == "__main__":
    pass