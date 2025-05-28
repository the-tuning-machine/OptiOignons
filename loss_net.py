import torch
from torch import nn
import torch.nn.functional as F


class LossNetworkBase(nn.Module):
    def __init__(self, output_dim=20, maximum=10):
        super(LossNetworkBase, self).__init__()
        self.maximum = maximum
        self.output_dim = output_dim
        self.register_buffer('linspace', torch.linspace(-self.maximum, self.maximum, output_dim))

    def distrib_to_loss(self, distribution):
        return (distribution @ self.linspace).mean()

    def loss_to_distrib(self, loss):
        clip_loss = torch.clip(loss, -self.maximum, self.maximum)
        p = (clip_loss + self.maximum) / (2 * self.maximum)
        p = p.unsqueeze(-1)
        binom = torch.distributions.Binomial(total_count=self.output_dim - 1, probs=p)
        k = torch.arange(self.output_dim)
        return torch.exp(binom.log_prob(k))

    def discretize(self, distribution, value):
        idx = torch.argmin(torch.abs(self.linspace - value))
        return distribution[idx]


class LossNetwork(LossNetworkBase):
    def __init__(self, input_dim, hidden_dim=16, output_dim=20, maximum=10):
        """
        Le réseau prend en entrée la sortie du student (de dimension input_dim)
        et produit un scalaire de perte. Pour introduire de la stochasticité (et
        pouvoir utiliser une mise à jour type REINFORCE), il renvoie également la
        log-probabilité associée.
        """
        super(LossNetwork, self).__init__(output_dim, maximum)
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


class HarmonicLoss(LossNetworkBase):
    def __init__(self, n=None, output_dim=20, maximum=20):
        super().__init__(output_dim, maximum)
        if n is None:
            self.n = nn.Parameter(torch.Tensor([1]))

    def forward(self, output, output_unembedding, target, target_unembedding):
        target_proba = self.proba(target, target_unembedding)
        output_proba = self.proba(output, output_unembedding)
        argmax = torch.argmax(target_proba, dim=1)
        
        # Use gather to select the probabilities at the same indices
        selected_output_proba = torch.gather(output_proba, 1, argmax.unsqueeze(1))
        
        log_loss = - torch.log(selected_output_proba.squeeze()).mean(dim=(0, 1))
        return self.loss_to_distrib(log_loss)

    def harmonic_mean(self, x, dim=0, epsilon=1e-8):
        return torch.mean((x + epsilon) ** (-self.n), dim=dim)

    def proba(self, embeddings, unembedding, epsilon=1e-8):
        # Fixed broadcasting: embeddings is (batch_size, seq_len, dim), unembedding is (vocab, dim)
        diff = unembedding[None, :, None, :] - embeddings[:, None, :, :]
        norm = torch.norm(diff, dim=-1)  # (batch_size, vocab, seq_len)
        return 1 / (norm + epsilon)**self.n * self.harmonic_mean(norm, epsilon=epsilon)
