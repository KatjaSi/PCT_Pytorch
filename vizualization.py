import numpy as np
import pyvista as pv
import scipy.spatial.distance
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from data_handling import parse_dataset, read_off, read_off_file, read_off_file2

def normilize(pointcloud):
        
    norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
    norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

    return  norm_pointcloud


def main():
    train_points, test_points, train_labels, test_labels, _ = parse_dataset(num_points=1024)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

   # x, y, z = train_points[0].T
    pointcloud = read_off_file2("/cluster/home/katjasi/PCT_Pytorch/data/ModelNet40/desk/train/desk_0015.off")
   # pointcloud = normilize(pointcloud)
    x, y, z = pointcloud.T
   # print(x)
    ax.scatter(x, y, z, c='b', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    print(len(x))
    plt.savefig("output.png")
    

if __name__ == '__main__':
    main()
