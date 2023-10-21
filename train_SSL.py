import torch
import torch.nn as nn
import numpy as np
import copy
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_metric_learning.losses import NTXentLoss
from pointSSL import POINT_SSL
from data_handling import  ModelNet, random_point_dropout, translate_pointcloud

from agumentation import random_volume_crop_pc

from data import load_data
import plotly.express as px


def train(model:POINT_SSL, train_loader:DataLoader, criterion,  optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model = model.half().to(device)
    model = nn.DataParallel(model)


    for epoch in range(num_epochs):
        running_loss = 0.0
        count = 0.0
        model.train()

        for x_prime, x in (train_loader):
            x_prime = x_prime.half().to(device)
            x = x.half().to(device)
            x = x.permute(0, 2, 1)
            x_prime = x_prime.permute(0,2,1)
            x_prime_rep, x_rep, x_prime_projection, x_projection = model(x_prime, x)
            embeddings = torch.cat((x_prime_projection, x_projection))
            print(embeddings.shape)




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
    # TODO: what should be for testing?

def main():
    train_points, _ = load_data("train")
    train_set = ModelNetForSSL(train_points, num_points=2048, crop_percentage=0.3)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1024, shuffle=True)

    model = POINT_SSL()

    loss = NTXentLoss(temperature = 0.1)
    optimizer = optim.Adam(lr=0.001, params=model.parameters())
    train(model, train_loader, loss, optimizer,1)



if __name__ == '__main__':
    main()