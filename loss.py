import torch
import torch.nn as nn


class LossNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        # On produit la moyenne et le log-écart-type pour la distribution normale
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x est supposé de taille [batch, 1]
        h = self.fc(x)
        mu = self.mean(h)
        log_std = self.log_std(h)
        std = torch.exp(log_std)
        # On définit la distribution
        dist = torch.distributions.Normal(mu, std)
        # On échantillonne une valeur à partir de cette distribution
        loss_value = (
            dist.rsample()
        )  # sample avec réparamétrisation (pour info, ici nous utiliserons REINFORCE)
        log_prob = dist.log_prob(loss_value)
        return loss_value.squeeze(), log_prob.squeeze()
