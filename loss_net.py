import torch
from torch import nn
import torch.nn.functional as F

class LossNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=20):
        """
        Le réseau prend en entrée la sortie du student (de dimension input_dim)
        et produit un scalaire de perte. Pour introduire de la stochasticité (et
        pouvoir utiliser une mise à jour type REINFORCE), il renvoie également la
        log-probabilité associée.
        """
        super(LossNetwork, self).__init__()
        self.maximum = 10
        self.output_dim = output_dim
        self.weights = torch.linspace(0, self.maximum, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, student_output, teacher_output):
        concat_output = torch.cat((student_output, teacher_output), dim=1)
        flat = concat_output.reshape(concat_output.size(0), -1)
        logits = self.mlp(flat).mean(dim=0)
        return F.softmax(logits, dim=-1)
    
    def distrib_to_loss(self, distribution):
        return torch.matmul(distribution, self.weights).mean()

    def loss_to_distrib(self, loss):
        p = loss / self.maximum
        binom = torch.distributions.Binomial(total_count=self.output_dim-1, probs=p)
        k = torch.arange(self.output_dim)
        return torch.exp(binom.log_prob(k))

    def discretize(self, distribution, value):
        idx = torch.argmin(torch.abs(self.weights - value))
        return distribution[idx]
   

class HarmonicLoss(nn.Module):
    def __init__(self, n=None):
        super().__init__()
        if n is None:
            self.n = nn.Parameter(torch.Tensor([1]))

    def forward(self, output, output_unembedding, target, target_unembedding):
        target_proba = self.proba(target, target_unembedding)
        argmax = torch.argmax(target_proba)
        output_proba = self.proba(output, output_unembedding)
        return -torch.log(output_proba[argmax])

    def harmonic_mean(self, x, dim, epsilon=1e-8):
        return 1 / torch.mean((x + epsilon) ** (-self.n), dim=dim)

    def proba(self, embeddings, unembedding):
        diff = unembedding[:, None, :] - embeddings[None, :, :]  # (vocab_size, seq_len, dim)
        norm = torch.norm(diff, dim=-1)  # (vocab_size, seq_len)
        # on fait le moyenne harmonic avec la puissance n
        return 1 / norm ** self.n * self.harmonic_mean(norm)