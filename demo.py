import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import plotly.express as px
from data_handling import uniform_sample_points

from data import load_data

def main():

    train_points, train_labels = load_data("train")

    pc = train_points[100]
    print(np.max(pc[:,2]))
    pc_upper, pc_down = agument_pc(pc, axis=1, num_points=2048)
    fig = px.scatter_3d(x = pc[:,0], y = pc[:,1], z = pc[:,2])
    fig.write_html('pc.html')
    fig = px.scatter_3d(x = pc_upper[:,0], y = pc_upper[:,1], z = pc_upper[:,2])
    fig.write_html('pc_upper.html')
    fig = px.scatter_3d(x = pc_down[:,0], y = pc_down[:,1], z = pc_down[:,2])
    fig.write_html('pc_down.html')


def agument_pc(pc, axis, num_points):
    pc_axis_projection = np.array(pc[:,axis])
    median =  np.median(pc_axis_projection)
    pc_upper = pc[pc_axis_projection  > median]
    pc_down = pc[pc_axis_projection < median]
    pc_upper = uniform_sample_points(pc_upper, num_points)
    pc_down = uniform_sample_points(pc_down, num_points)
    # back to take the same space volume as before
    # scale
    pc_max = np.max(pc[:,axis])
    pc_min = np.min(pc[:,axis])
    
    __scale_pc__(pc_upper, axis, pc_min, pc_max)
    __scale_pc__(pc_down, axis, pc_min, pc_max)
    return pc_upper, pc_down

def __scale_pc__(pc, axis, new_min, new_max):
    pc_max = np.max(pc[:,axis])
    pc_min = np.min(pc[:,axis])
    scale_factor = (new_max-new_min)/(pc_max-pc_min)
    pc[:,axis] = pc[:,axis]*scale_factor #+ new_min + new_max-np.min(pc[:,axis])-np.max(pc[:,axis])
    pc[:,axis] = pc[:,axis]+new_min-np.min(pc[:,axis])

if __name__ == '__main__':
    main()