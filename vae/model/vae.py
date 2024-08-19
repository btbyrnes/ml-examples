import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_ = nn.Linear(hidden_dim, latent_dim)
        self.logVar_ = nn.Linear(hidden_dim, latent_dim)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor]:
        h = self.activation( self.linear1(x) )
        h = self.activation( self.linear2(h) )
        mu = self.mu_(h)
        logVar = self.logVar_(h)

        return mu, logVar
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.LeakyReLU(0.2)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        h = self.activation( self.linear1(x) )
        h = self.activation( self.linear2(h) )
        x_hat = torch.sigmoid( self.output(h) )

        return x_hat


class AutoEncoder(nn.Module):
    def __init__(self, enc:nn.Module, dec:nn.Module) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = enc 
        self.decoder = dec

    def reparameterization(self, mean, var) -> torch.Tensor:
        eps = torch.randn_like(var)
        z = mean + var * eps
        return z

    def forward(self, X:torch.Tensor) -> tuple[torch.Tensor]:
        mu, logVar = self.encoder(X)
        z = self.reparameterization(mu, torch.exp(0.5 * logVar))
        x_hat = self.decoder(z)

        return x_hat, mu, logVar 