import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from data_handling import read_off, read_off_files_in_folder, uniform_sample_points, parse_dataset, ModelNet
from model import PCT, SPCT
from train import train
import torch.nn.functional as F
import torch.optim as optim
from criterion import cross_entropy_loss_with_label_smoothing

import plotly.express as px

from data import load_data

def main():
    pct = SPCT(output_channels=40)

    train_points, train_labels = load_data("train")
    test_points, test_labels = load_data("test")
    #print(train_points[0][:,2].size)

    pc = train_points[0]
    print(np.max(pc[:,2]))
    pc_upper, pc_down = agument_pc(pc, axis=1, num_points=2048)
    print(np.max(pc_upper[:,2]))
    print(np.max(pc_down[:,2]))
   # fig = px.scatter_3d(x = pc_upper[:,0], y = pc_upper[:,1], z = pc_upper[:,2])
   # fig.write_html('pc_upper.html')
   # fig = px.scatter_3d(x = pc_down[:,0], y = pc_down[:,1], z = pc_down[:,2])
    #fig.write_html('pc_down.html')


def agument_pc(pc, axis, num_points):
    pc_axis_projection = np.array(pc[:,axis])
    median =  np.median(pc_axis_projection)
    pc_upper = pc[pc_axis_projection  > median]
    pc_down = pc[pc_axis_projection < median]
    pc_upper = uniform_sample_points(pc_upper, num_points)
    pc_down = uniform_sample_points(pc_down, num_points)
    # back to take the same space volume as before

    #x_max = np.max(pc[:,axis])
    #x_min = np.min(pc[:,axis])
    #pc_upper[:,axis] = pc_upper[:,axis]*(x_max-x_min)+x_min
    #pc_down[:,axis] = pc_down[:,axis]*(x_max-x_min)+x_min
    return pc_upper, pc_down

if __name__ == '__main__':
    main()