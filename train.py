import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import PCT

def train(model:PCT, train_loader:DataLoader, criterion, optimizer, num_epochs, device="cpu"):
    device = torch.device(device)
    model = model.double()
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for data, labels in (train_loader):
            #batch_size = data.size()[0]
            data = data.permute(0, 2, 1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"epoch {epoch}: loss: {loss.item()}")
    
    print("Finished Training")

