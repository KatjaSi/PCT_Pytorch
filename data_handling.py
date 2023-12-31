import glob
import os
import numpy as np
import multiprocessing
from torch.utils.data import Dataset

from data import load_data

DATA_DIR_10 = "data/ModelNet10" 
DATA_DIR_40 = "data/ModelNet40" 

def read_off(file):
    header_line = file.readline().strip()
    if not header_line.startswith('OFF'):
        # Check if the header line starts with 'OFF'
        raise Exception(f"Not a valid OFF header for file {file.name}")
    
    # If the header is split across two lines
    if len(header_line.split()) > 1:
        header_parts = header_line.split()
        n_verts = int(header_parts[-3][3:])
        n_faces = int(header_parts[-2])
        n_edges = int(header_parts[-1])
    else:
        n_verts, n_faces, n_edges = map(int, file.readline().strip().split(' '))
    
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    return np.array(verts)

def read_off2(file):
    off_header = file.readline().strip()
    if 'OFF' == off_header:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    return np.array(verts)


def read_off_file(file_path):
    with open(file_path) as file:
        verts = read_off(file)
    return verts

def read_off_file2(file_path):
    with open(file_path) as file:
        verts = read_off2(file)
    return verts

def read_off_files_in_folder(folder_path):
    off_files = [f for f in os.listdir(folder_path) if f.endswith(".off")]

    all_verts = []
    pool = multiprocessing.Pool()  # Create a pool of worker processes

    file_paths = [os.path.join(folder_path, off_file) for off_file in off_files]
    all_verts = pool.map(read_off_file, file_paths)  # Use map to apply read_off_file to each file path
    
    pool.close()
    pool.join()
    return all_verts

def uniform_sample_points(pointcloud, num_points):
    if len(pointcloud) >= num_points:
        sampled_indices = np.random.choice(len(pointcloud), num_points, replace=False)
    else:
        sampled_indices = np.random.choice(len(pointcloud), num_points, replace=True)
    return pointcloud[sampled_indices]

def parse_dataset(dataset="modelnet10", num_points=1024, cached=True):

    # returned cached if saved
    saved_data_dir = f"data/saved_sampled_points/{dataset}/{num_points}"
    if not os.path.exists(saved_data_dir):
        os.makedirs(saved_data_dir)
    train_points_file = f"{saved_data_dir}/train_points.npy"
    test_points_file = f"{saved_data_dir}/test_points.npy"
    train_labels_file = f"{saved_data_dir}/train_labels.npy"
    test_labels_file = f"{saved_data_dir}/test_labels.npy"
    class_map_file = f"{saved_data_dir}/class_map.npy"
    if (cached  and os.path.exists(train_points_file) and os.path.exists(test_points_file) and \
       os.path.exists(train_labels_file) and os.path.exists(test_labels_file) and \
       os.path.exists(class_map_file) ):
        train_points = np.load(train_points_file)
        test_points = np.load(test_points_file)
        train_labels = np.load(train_labels_file)
        test_labels = np.load(test_labels_file)
        class_map = np.load(class_map_file, allow_pickle=True).item()
        return     (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )
        

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}


    data_dir = DATA_DIR_10 if dataset=="modelnet10" else DATA_DIR_40

    folders = [folder for folder in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(folder)]
    print(f"datadir: {data_dir}")
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

    # saving
    np.save(train_points_file, np.array(train_points))
    np.save(test_points_file, np.array(test_points))
    np.save(train_labels_file, np.array(train_labels))
    np.save(test_labels_file, np.array(test_labels))
    np.save(class_map_file, class_map)

    return     (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ModelNet(Dataset):
    def __init__(self, data, labels, num_points=1024, set_type="train"):
        self.data = data
        self.labels = labels
        self.set_type = set_type
        self.num_points = num_points

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx): # TODO: check how the data should be agumented
        x = self.data[idx][:self.num_points]
        y = self.labels[idx]
        if self.set_type == 'train':
            x = random_point_dropout(x) 
            x = translate_pointcloud(x)
            np.random.shuffle(x)
            # TODO: anisotropic scaling?
        return x, y    

def main():
    train_points, test_points, train_labels, test_labels, _ = parse_dataset(num_points=1024)
    train_set = ModelNet(train_points, train_labels)
    test_set = ModelNet(test_points, test_labels)
    for data, label in test_set:
        print(data) 
        #print(label)
       # print(data) #data is not normilized
    

if __name__ == '__main__':
    main()