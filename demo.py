import torch
from torch import tensor
from torch.utils.data import DataLoader
from data_handling import read_off, read_off_files_in_folder, uniform_sample_points, parse_dataset, ModelNet10
from model import PCT
from train import train
import torch.nn.functional as F
import torch.optim as optim
from criterion import cross_entropy_loss_with_label_smoothing

def main():
    #file = open("data/ModelNet10/bed/train/bed_0001.off")
    #verts_1 = read_off(file)
    #
    #file.close()
    
    # Convert verts_1 to a PyTorch tensor
  #  all_verts = read_off_files_in_folder("data/ModelNet10/bed/train")
   # all_verts2 = []
    #for pointcloud in all_verts:
     #   all_verts2.append(uniform_sample_points(pointcloud, 1024))
    #verts_tensor = tensor(all_verts2, dtype=torch.float32)
    #verts_tensor = DataLoader(verts_tensor, batch_size=32, shuffle=True)
    
    #pc_tensor = verts_tensor.unsqueeze(0)
    # Print the resulting PyTorch tensor
    #print(pc_tensor.permute(0, 2, 1).size())

    pct = PCT()
   # output = pct(pc_tensor.permute(0, 2, 1))
    train_points, test_points, train_labels, test_labels, _ = parse_dataset(num_points=1024)
    #output = pct(verts_tensor[:32].permute(0, 2, 1))
   # output = pct(tensor(test_points[:32], dtype=torch.float32).permute(0, 2, 1))
   # print(output)

    train_set = ModelNet10(train_points, train_labels)
    test_set = ModelNet10(test_points, test_labels)

    # Set batch size
    batch_size = 32

    # Create DataLoader instances
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

#TODO: change criterion
    opt = optim.SGD(pct.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # optim.Adam(lr=0.0001, params=pct.parameters(
    train(pct, train_loader, cross_entropy_loss_with_label_smoothing, opt, 32)


if __name__ == '__main__':
    main()