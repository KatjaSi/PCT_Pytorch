import glob
import os
import numpy as np
from torch.utils.data import Dataset

DATA_DIR = "data/ModelNet10" 

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, _, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    return np.array(verts)

def read_off_files_in_folder(folder_path):
    off_files = [f for f in os.listdir(folder_path) if f.endswith(".off")]

    all_verts = []
    for off_file in off_files:
        file_path = os.path.join(folder_path, off_file)
        with open(file_path) as file:
            verts = read_off(file)
            all_verts.append(verts)
    
    return all_verts

def uniform_sample_points(pointcloud, num_points):
    if len(pointcloud) >= num_points:
        sampled_indices = np.random.choice(len(pointcloud), num_points, replace=False)
        sampled_pointcloud = pointcloud[sampled_indices]
    else:
        sampled_indices = np.random.choice(len(pointcloud), num_points, replace=True)
        sampled_pointcloud = pointcloud[sampled_indices]
    return sampled_pointcloud

def parse_dataset(num_points=1024):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}

    folders = [folder for folder in glob.glob(os.path.join(DATA_DIR, "*")) if os.path.isdir(folder)]
    print(f"datadir: {DATA_DIR}")
    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))

        class_map[i] = folder.split("\\")[-1]

         # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            with open(f) as file:
                pointcloud = read_off(file)
                train_points.append(uniform_sample_points(pointcloud, num_points))
            train_labels.append(np.int64(i))

        for f in test_files:
            with open(f) as file:
                pointcloud = read_off(file)
                test_points.append(uniform_sample_points(pointcloud, num_points))
            test_labels.append(np.int64(i))

    return     (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

class ModelNet10(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y    


def main():
    
    #file = open("data/ModelNet10/bed/train/bed_0001.off")
    #verts_1 = read_off(file)
    #file.close()
   # all_verts = read_off_files_in_folder("data/ModelNet10/bed/train")
   # print(uniform_sample_points(all_verts[0], 10))
    _, _, train_labels, _, _ = parse_dataset()
    print(train_labels)
    

if __name__ == '__main__':
    main()