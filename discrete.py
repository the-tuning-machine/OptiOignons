import torch
import torch.nn as nn
import torch.optim as optim
from llama_cpu import Transformer, ModelArgs  # On suppose que ce module est disponible
from utils import freeze, unfreeze, weight_mse
from loss_net import LossNetwork, HarmonicLoss, CrossEntropyLoss, MSELoss
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)

vocab_size = 32
dim = 4
seq_len = 100
batch_size = 16
params = ModelArgs(dim, 2, 2, vocab_size=vocab_size)

# loss_net = LossNetwork(input_dim=2*dim*seq_len, hidden_dim=dim)
# loss_net = HarmonicLoss()
# loss_net = CrossEntropyLoss()
loss_net = MSELoss()
loss_net_optimizer = optim.Adam(loss_net.parameters(), lr=0.001)
# loss_net_optimizer = optim.Adam(loss_net.parameters())
num_iterations = 100000
num_episode = 100
eps = 1e-3

writer = SummaryWriter(log_dir="runs/discrete_experiment")

for episode in range(num_episode):
    teacher = Transformer(params)
    student = Transformer(params)
    freeze(teacher)
    student_optimizer = optim.AdamW(student.parameters(), weight_decay=0.1, lr=0.001)
    # student_optimizer = optim.AdamW(student.parameters())

    for iteration in range(num_iterations):
        # print(100 * iteration / num_iterations, "%")
        old_weight__mse = weight_mse(teacher, student)
        # print("Distance:", old_weight__mse)

        student_optimizer.zero_grad()
        X = torch.randint(0, vocab_size, (batch_size, seq_len))
        teacher_output, teacher_unembedding = teacher(X, 0).detach(), teacher.output.weight
        student_output, student_unemebedding = student(X, 0), student.output.weight

        loss_distrib_output = loss_net(
            student_output, student_unemebedding, teacher_output, teacher_unembedding
        )
        loss_output = loss_net.distrib_to_loss(loss_distrib_output)
        loss_output.backward(retain_graph=True)
        student_optimizer.step()

        # on train la loss en fonction de la reward
        loss_net_optimizer.zero_grad()
        new_weight__mse = weight_mse(teacher, student)
        reward = torch.clip(old_weight__mse - new_weight__mse, -0.1, 0.1)
        loss_output = loss_net.distrib_to_loss(loss_distrib_output)
        loss_loss: torch.Tensor = -reward * loss_net.discretize(
            loss_distrib_output, loss_output
        )
        # print("Reward:", reward)
        # loss_loss.backward()
        grads = torch.autograd.grad(
            outputs=loss_loss,
            inputs=loss_net.parameters(),
            retain_graph=False,
            allow_unused=True,
        )

        for p, g in zip(loss_net.parameters(), grads):
            p.grad = g

        loss_net_optimizer.step()
        # print("Loss loss:", loss_loss)
        # print("Loss output", loss_output)
        # print()

        writer.add_scalar("Distance/weight_mse", old_weight__mse, iteration)
        writer.add_scalar("Reward", reward, iteration)
        writer.add_scalar("Loss/loss_loss", loss_loss.item(), iteration)
        writer.add_scalar("Loss/loss_output", loss_output.item(), iteration)

print("Entraînement terminé.")
writer.close()
