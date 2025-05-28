import torch
import torch.nn as nn
import torch.optim as optim
from llama_cpu import Transformer, ModelArgs  # On suppose que ce module est disponible
import copy

CLASSROOM_SIZE = 3


# =============================================================================
# Définition du Loss Network (L) sous forme d'une politique stochastique
# =============================================================================
class LossNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        """
        Le réseau prend en entrée la sortie du student (de dimension input_dim)
        et produit un scalaire de perte. Pour introduire de la stochasticité (et
        pouvoir utiliser une mise à jour type REINFORCE), il renvoie également la
        log-probabilité associée.
        """
        super(LossNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, student_output):
        flat = student_output.reshape(student_output.size(0), -1).clone()
        return self.mlp(flat).mean()


def freeze(model):
    for i, param in enumerate(model.parameters()):
        if i % 2 == 1:
            param.requires_grad = False


def detach(model):
    for param in model.parameters():
        param.detach()


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


# =============================================================================
# Fonction pour calculer la MSE entre les poids du teacher et ceux du student
# =============================================================================
def weight_mse(teacher, student):
    mse = 0.0
    for p_teacher, p_student in zip(teacher.parameters(), student.parameters()):
        mse += torch.mean((p_teacher - p_student) ** 2)
    return mse


# =============================================================================
# Instanciation des trois modèles
# =============================================================================
vocab_size = 10
dim = 16
seq_len = 32
batch_size = 4
params = ModelArgs(dim, 2, 2, vocab_size=vocab_size)

# Création du modèle teacher (on ne le mettra jamais à jour)
# Création du modèle student identique

# Pour le loss network, la dimension d'entrée est celle de la sortie du student.
# Ici, nous supposons que la sortie du Transformer a une dimension fixée (par exemple, 32).
loss_nets = [
    LossNetwork(input_dim=2 * dim * seq_len, hidden_dim=dim)
    for _ in range(CLASSROOM_SIZE)
]
loss_net_optimizers = [optim.Adam(loss_net.parameters()) for loss_net in loss_nets]

num_iterations = 10000
num_episode = 100
eps = 1e-3

student_clone = Transformer(params)

for episode in range(num_episode):
    teacher = Transformer(params)
    for param in teacher.parameters():
        param.requires_grad = False
    student = Transformer(params)
    student_optimizer = optim.AdamW(student.parameters())
    for iteration in range(num_iterations):
        X = torch.randint(0, vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            teacher_output = teacher(X, 0)

        student_optimizer.zero_grad()
        student_output = student(X, 0)

        student_clone.load_state_dict(student.state_dict())
        loss_outputs = [None for _ in range(CLASSROOM_SIZE)]
        loss_errors = [None for _ in range(CLASSROOM_SIZE)]

        for classmate in range(CLASSROOM_SIZE):
            loss_net = loss_nets[classmate]
            loss_net_optimizers[classmate].zero_grad()

            concat_output = torch.cat((student_output, teacher_output), dim=1)
            loss_outputs[classmate] = loss_net(concat_output)
            freeze(loss_net)
            loss_outputs[classmate].backward(retain_graph=True)
            student_optimizer.step()
            unfreeze(loss_net)
            loss_errors[classmate] = weight_mse(teacher, student)

        freeze(student)
        best_classmate = loss_errors.index(min(loss_errors))
        y = loss_outputs[best_classmate]
        loss_losses = [None for _ in range(CLASSROOM_SIZE)]

        for classmate in range(CLASSROOM_SIZE):
            loss_losses[classmate] = (loss_errors[classmate] - y) ** 2
            loss_losses[classmate].backward()
            loss_net_optimizers[classmate].step()

        unfreeze(student)

print("Entraînement terminé.")
