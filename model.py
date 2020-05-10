import torch
from torch import nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self, channel, n_hidden, z_dim):
        super(VariationalAutoEncoder, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(channel, n_hidden, z_dim)
        self.decoder = Decoder(channel, n_hidden, z_dim)

    def z_sample(self, mean, logvar):
        eps = torch.randn(mean.shape).to(self.device)
        return mean + torch.exp(0.5*logvar) * eps

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.z_sample(mean, logvar)
        x_hat = self.decoder(z)
        return mean, logvar, z, x_hat

    def inverse(self, z):
        x_hat = self.decoder(z)
        return x_hat


class Encoder(nn.Module):
    def __init__(self, in_ch, n_hidden, z_dim):
        super(Encoder, self).__init__()
        self.enc = nn.Linear(in_ch, n_hidden)
        self.mu = nn.Linear(n_hidden, z_dim)
        self.sigma = nn.Linear(n_hidden, z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.relu(self.enc(x))
        mean = self.mu(out)
        logvar = self.softplus(self.sigma(out))
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, out_ch, n_hidden, z_dim):
        super(Decoder, self).__init__()
        self.dec1 = nn.Linear(z_dim, n_hidden)
        self.dec2 = nn.Linear(n_hidden, out_ch)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.relu(self.dec1(z))
        out = self.sigmoid(self.dec2(out))
        return out

