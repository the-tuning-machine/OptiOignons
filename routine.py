import torch
import torch.nn as nn
import torch.optim as optim
from llama_cpu import Transformer, ModelArgs  # On suppose que ce module est disponible

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
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(self, student_output):
        flat = student_output.flatten().clone()
        mu, sigma = self.mlp(flat)
        sigma = torch.exp(sigma)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        # return action.item(), action_dist.log_prob(action)
        return mu, action_dist.log_prob(action)
    
def freeze(model):
    for i, param in enumerate(model.parameters()):
        if i%2==1:
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
params = ModelArgs(
    16, 
    2, 
    2,
    vocab_size=vocab_size
)


# Création du modèle teacher (on ne le mettra jamais à jour)
teacher = Transformer(params)
# Création du modèle student identique
student = Transformer(params)

# Pour le loss network, la dimension d'entrée est celle de la sortie du student.
# Ici, nous supposons que la sortie du Transformer a une dimension fixée (par exemple, 32).
output_dim = 1280  # À adapter selon votre modèle Transformer
loss_net = LossNetwork(input_dim=output_dim, hidden_dim=16)

# On fige les poids du teacher (et on utilisera aussi une phase où L est gelé)
for param in teacher.parameters():
    param.requires_grad = False

# =============================================================================
# Définition des optimisateurs pour le student et pour le loss network
# =============================================================================
student_optimizer = optim.AdamW(student.parameters())
loss_net_optimizer = optim.Adam(loss_net.parameters())

# =============================================================================
# Boucle d'entraînement
# =============================================================================
num_iterations = 10000
batch_size = 4

for iteration in range(num_iterations):
    # --- Génération d'un batch d'exemples ---
    # Exemple d'entrées aléatoires de dimension [batch_size, sequence_dim]
    # Ici, sequence_dim est arbitraire (par exemple, 32)
    X = torch.randint(0, vocab_size, (batch_size, 32))
    
    # Passage dans le teacher pour obtenir les sorties "cibles"
    with torch.no_grad():
        teacher_output = teacher(X, 0)  # [batch_size, output_dim]
    
    # Passage dans le student pour obtenir la prédiction
    student_out = student(X, 0)
    action, log_prob = loss_net(student_out)  # [batch_size, output_dim]
    student_optimizer.zero_grad()
    freeze(loss_net)
    action.backward(retain_graph=True)
    student_optimizer.step()
    unfreeze(loss_net)
    freeze(student)
    pen = weight_mse(teacher, student)
    loss_loss = pen * log_prob
    loss_net_optimizer.zero_grad()
    loss_loss.backward()
    loss_net_optimizer.step()
    unfreeze(student)
    
    # --- Mise à jour du loss network ---
    # Après la mise à jour du student, on calcule la MSE entre les poids de teacher et student
    # L'idée est que plus la MSE est faible, mieux le loss network a guidé student.
    # Nous définissons ainsi un signal de type "reward" pour L : ici, le reward est négatif
    # lorsque la MSE est élevée.
    # reward = -teacher_student_mse.detach()
    
    # Pour une mise à jour de type REINFORCE, la loss pour L est :
    #   loss_L = - log_prob * reward
    # loss_L = -reward
    # loss_net_optimizer.zero_grad()
    # loss_L.backward()
    # loss_net_optimizer.step()
    
    # Affichage de quelques informations
    if iteration % 100 == 0:
        # print(f"Iteration {iteration} | Student loss: {student_loss.item():.4f} "
            #   f"| Teacher-Student MSE: {teacher_student_mse.item():.4f} "
            #   f"| Reward: {reward.item():.4f}"
        # )
        print(
            f"Iteration {iteration}, {pen}, {student.tok_embeddings.weight}"
        )

print("Entraînement terminé.")