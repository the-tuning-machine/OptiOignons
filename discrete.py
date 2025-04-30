import torch
import torch.nn as nn
import torch.optim as optim
from llama_cpu import Transformer, ModelArgs  # On suppose que ce module est disponible
from loss_net import LossNetwork
from utils import freeze, unfreeze, weight_mse
torch.autograd.set_detect_anomaly(True)

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

loss_net = LossNetwork(input_dim=2*dim*seq_len, hidden_dim=dim)
loss_net_optimizer = optim.Adam(loss_net.parameters(), lr=1)
# loss_net_optimizer = optim.Adam(loss_net.parameters())
num_iterations = 10000
num_episode = 100
eps = 1e-3

for episode in range(num_episode):
    teacher = Transformer(params)
    student = Transformer(params)
    freeze(teacher)
    student_optimizer = optim.AdamW(student.parameters(), lr=1)
    # student_optimizer = optim.AdamW(student.parameters())

    for iteration in range(num_iterations):
        print(iteration/num_iterations)
        old_weight__mse = weight_mse(teacher, student)
        print("Distance:", old_weight__mse)

        student_optimizer.zero_grad()
        X = torch.randint(0, vocab_size, (batch_size, seq_len))
        teacher_output = teacher(X, 0).detach()
        student_output = student(X, 0)

        loss_distrib_output = loss_net(student_output, teacher_output)
        loss_output = loss_net.distrib_to_loss(loss_distrib_output)
        loss_output.backward(retain_graph=True)
        student_optimizer.step()

        # on train la loss en fonction de la reward
        loss_net_optimizer.zero_grad()
        new_weight__mse = weight_mse(teacher, student)
        reward = old_weight__mse - new_weight__mse
        loss_output = loss_net.distrib_to_loss(loss_distrib_output)
        loss_loss: torch.Tensor = - reward * loss_net.discretize(loss_distrib_output, loss_output)
        print("reward", reward)

        # loss_loss.backward()
        grads = torch.autograd.grad(
            outputs=loss_loss,
            inputs=loss_net.parameters(),
            retain_graph=False,
            allow_unused=True
        )

        for p, g in zip(loss_net.parameters(), grads):
            p.grad = g

        loss_net_optimizer.step()
        print(loss_loss)

print("Entraînement terminé.")