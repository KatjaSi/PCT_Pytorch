import glob
import os
import numpy as np
import multiprocessing
from torch.utils.data import Dataset

DATA_DIR_10 = "data/ModelNet10" 
DATA_DIR_40 = "data/ModelNet40" 

#def read_off(file):
 #   if 'OFF' != file.readline().strip():
  #      raise Exception(f"Not a valid OFF header for file{file.name}")
   # n_verts, _, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
    #verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    #return np.array(verts)
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


def read_off_file(file_path):
    with open(file_path) as file:
        verts = read_off(file)
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

class ModelNet(Dataset):
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