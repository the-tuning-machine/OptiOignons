import torch
import torch.nn as nn
import torch.optim as optim

# On crée un modèle A (teacher) : A_in -> A_out
# On crée un modèle B (student) identique : A_in -> A_out
# On crée un modèle L (loss) : A_out -> 1
# On passe génèren des données
# On les passe dans A
# On les passe dans B
# On passe la sortie de B dans L
# On freeze les poids de L
# On update les poids de B avec la loss L (sans mettre à jour ceux de L)
# On calcule la MSE entre les poids de A et ceux de B et cela devient la loss de L
# On 




# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Boucle de test
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%)\n')

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Entraîner et tester le modèle
for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)