import torch
from torch import tensor
from torch.utils.data import DataLoader
from data_handling import read_off, read_off_files_in_folder, uniform_sample_points, parse_dataset, ModelNet
from model import PCT, SPCT
from train import train
import torch.nn.functional as F
import torch.optim as optim
from criterion import cross_entropy_loss_with_label_smoothing

from data import load_data

def main():
    pct = SPCT(output_channels=40)

    #train_points, test_points, train_labels, test_labels, _ = parse_dataset(dataset="modelnet10", num_points=1024)

    train_points, train_labels = load_data("train")
    test_points, test_labels = load_data("test")

    train_set = ModelNet(train_points, train_labels)
    test_set = ModelNet(test_points, test_labels)

    # Set batch size
    batch_size = 32

    # Create DataLoader instances
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=batch_size, shuffle=False)

    opt = optim.SGD(pct.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4) # optim.Adam(lr=0.0001, params=pct.parameters(
    train(  model=pct,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=cross_entropy_loss_with_label_smoothing,
            optimizer=opt,
            num_epochs=100
            )


    #TODO: display 3d array


if __name__ == '__main__':
    main()