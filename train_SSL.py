import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pointSSL import POINT_SSL
from data_handling import  ModelNet, random_point_dropout, translate_pointcloud

from agumentation import random_volume_crop_pc

from data import load_data
import plotly.express as px


def train(model:POINT_SSL, train_loader:DataLoader, criterion,  num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model = model.half().to(device)
    model = nn.DataParallel(model)


    for epoch in range(num_epochs):
        running_loss = 0.0
        count = 0.0
        model.train()

        train_pred = []
        train_true = []
        idx = 0
        for data, labels in (train_loader):
            batch_size = len(labels)
            data = data.half().to(device)  # Move data to device
            labels = labels.to(device)
            data = data.permute(0, 2, 1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = outputs.max(dim=1)[1]
            count += batch_size
            running_loss += loss.item() * batch_size
            train_true.append(labels.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1
            
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Epoch: %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                    running_loss*1.0/count,
                                                                                    metrics.accuracy_score(
                                                                                    train_true, train_pred),
                                                                                    metrics.balanced_accuracy_score(
                                                                                    train_true, train_pred))


class ModelNetForSSL(Dataset):
    def __init__(self, data, num_points=1024, crop_percentage=0.4):
        self.data = data
        self.num_points = num_points
        self.crop_percentage = crop_percentage

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        x = self.data[idx][:self.num_points]
        x_prime = copy.deepcopy(x)

        x = random_point_dropout(x) 
        x = translate_pointcloud(x)
        np.random.shuffle(x)

        x_prime = random_volume_crop_pc(x_prime, crop_percentage=self.crop_percentage)
        x_prime = random_point_dropout(x_prime) 
        x_prime = translate_pointcloud(x_prime)
        np.random.shuffle(x_prime)

        return x_prime, x

def main():
    train_points, _ = load_data("train")
    train_set = ModelNetForSSL(train_points, num_points=2048, crop_percentage=0.3)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1024, shuffle=True)

    for x_prime, x in (train_loader):
        pc = x_prime[1]
        fig = px.scatter_3d(x = pc[:,0], y = pc[:,1], z = pc[:,2])
        fig.write_html('pc_prime.html')

        pc = x[1]
        fig = px.scatter_3d(x = pc[:,0], y = pc[:,1], z = pc[:,2])
        fig.write_html('pc.html')

        break



if __name__ == '__main__':
    main()