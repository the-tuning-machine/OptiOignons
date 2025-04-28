import torch
import torch.nn as nn
import torch.optim as optim
from llama_cpu import Transformer, ModelArgs  # On suppose que ce module est disponible
import copy
import torch.nn.functional as F
from torch.autograd.grad_mode import _unsafe_preserve_version_counter

torch.autograd.set_detect_anomaly(True)

CLASSROOM_SIZE = 3
# =============================================================================
# Définition du Loss Network (L) sous forme d'une politique stochastique
# =============================================================================
class LossNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=20):
        """
        Le réseau prend en entrée la sortie du student (de dimension input_dim)
        et produit un scalaire de perte. Pour introduire de la stochasticité (et
        pouvoir utiliser une mise à jour type REINFORCE), il renvoie également la
        log-probabilité associée.
        """
        super(LossNetwork, self).__init__()
        self.weights = torch.linspace(0, 10, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, student_output):
        flat = student_output.reshape(student_output.size(0), -1).clone()
        logits = self.mlp(flat).mean(dim=0)
        return F.softmax(logits, dim=-1)
    
    def distrib_to_loss(self, distribution):
        return torch.matmul(distribution, self.weights).mean()
    
    def discretize(self, distribution, value):
        idx = torch.argmin(torch.abs(self.weights - value))
        return distribution[idx]
    
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

def zeroed_versions(model):
    for param in model.parameters():
        param._version = 0

class preserve_version_context:
    def __init__(self, model):
        self.model = model
        self.saved_versions = {}

    def __enter__(self):
        for param in self.model.parameters():
            if param.requires_grad:
                self.saved_versions[param] = param._version

    def __exit__(self, exc_type, exc_value, traceback):
        for param, saved_version in self.saved_versions.items():
            with _unsafe_preserve_version_counter(param):
                torch._C._autograd._unsafe_set_version_counter(param, saved_version)

def get_versions(model):
    for name, param in model.named_parameters():
        if param._version != 0:
            print(f"Pas à la version de base {name}, {param._version}")
        else:
            print(f"À la version de base {name}, {param._version}")

# =============================================================================
# Fonction pour calculer la MSE entre les poids du teacher et ceux du student
# =============================================================================
def weight_mse(teacher, student):
    mse = 0.0
    for p_teacher, p_student in zip(teacher.parameters(), student.parameters()):
        mse += torch.mean((p_teacher - p_student) ** 2)
    return mse.detach()

# =============================================================================
# Instanciation des trois modèles
# =============================================================================
vocab_size = 10
dim = 16
seq_len = 32
batch_size = 4
params = ModelArgs(
    dim, 
    2, 
    2,
    vocab_size=vocab_size
)

# Création du modèle teacher (on ne le mettra jamais à jour)
# Création du modèle student identique

# Pour le loss network, la dimension d'entrée est celle de la sortie du student.
# Ici, nous supposons que la sortie du Transformer a une dimension fixée (par exemple, 32).
loss_net = LossNetwork(input_dim=2*dim*seq_len, hidden_dim=dim)
# loss_net_optimizer = optim.Adam(loss_net.parameters(), lr=1)
loss_net_optimizer = optim.Adam(loss_net.parameters())
num_iterations = 10000
num_episode = 100
eps = 1e-3

student_clone = Transformer(params)

for episode in range(num_episode):
    teacher = Transformer(params)
    student = Transformer(params)
    freeze(teacher)
    # student_optimizer = optim.AdamW(student.parameters(), lr=1)
    student_optimizer = optim.AdamW(student.parameters())

    for iteration in range(num_iterations):
        with torch.no_grad():
            old_weight__mse = weight_mse(teacher, student)

        student_optimizer.zero_grad()
        X = torch.randint(0, vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            teacher_output = teacher(X, 0).detach()
        student_output = student(X, 0)
        concat_output = torch.cat((student_output, teacher_output), dim=1)
        concat_output_clone = concat_output.detach().clone()

        freeze(loss_net)
        loss_distrib_output = loss_net(concat_output)
        loss_output = loss_net.distrib_to_loss(loss_distrib_output)
        loss_output.backward()
        student_optimizer.step()
        unfreeze(loss_net)

        with torch.no_grad():
            new_weight__mse = weight_mse(teacher, student)
        # reward = old_weight__mse - new_weight__mse
        reward = (old_weight__mse - new_weight__mse).detach()

        # on train la loss en fonction de la reward
        loss_net_optimizer.zero_grad()
        loss_distrib_output = loss_net(concat_output_clone)
        loss_output = loss_net.distrib_to_loss(loss_distrib_output)
        loss_loss: torch.Tensor = - reward * loss_net.discretize(loss_distrib_output, loss_output)
        freeze(student)
        loss_loss.backward()
        loss_net_optimizer.step()
        unfreeze(student)
        print(loss_loss)

print("Entraînement terminé.")