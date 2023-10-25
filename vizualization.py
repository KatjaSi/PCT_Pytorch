import numpy as np
import pyvista as pv
import scipy.spatial.distance
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE 
import seaborn as sns 
import pandas as pd

from pointSSL import POINT_SSL
from data import load_data
from data_handling import  ModelNet


def normilize(pointcloud):
        
    norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
    norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

    return  norm_pointcloud


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    point_ssl = POINT_SSL()
    point_ssl.load_state_dict(torch.load('checkpoints/models/point_ssl_1000.t7'), strict=False)

    train_points, train_labels = load_data("test")

    # Get indices of samples with labels in the first 5 classes
    indices_first_5_classes = np.where(train_labels < 10)[0]

    # Filter train_points and train_labels to only keep 5 first classes
    train_points = np.array([train_points[i] for i in indices_first_5_classes])
    train_labels = np.array([train_labels[i] for i in indices_first_5_classes])

    # get sample batch
    train_set = ModelNet(train_points, train_labels, set_type="test", num_points=2048) #TODO: replace with test
    loader = DataLoader(dataset=train_set, num_workers=2, batch_size=256, shuffle=True)   
    sample = next(iter(loader))
    data, labels = sample

    labels = labels.cpu().detach().numpy()
    labels = labels.reshape(-1)


    # get representations
    point_ssl = point_ssl.half().to(device)
    point_ssl.eval()
    data = data.half().to(device)
    data = data.permute(0, 2, 1)
    embeddings, _ = point_ssl(data)
    embeddings = embeddings.cpu().detach().numpy()

    # get low dims tsne embeddings
    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)

    # Plot
    ax =sns.scatterplot(x=embeddings_2d[:,0], y=embeddings_2d[:,1], alpha=0.5, hue=labels, palette="tab10")

    plt.savefig('scatter_plot_test.png')


    

 
    

if __name__ == '__main__':
    main()
