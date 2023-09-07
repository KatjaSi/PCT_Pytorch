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
    pct = PCT()

    train_points, test_points, train_labels, test_labels, _ = parse_dataset(num_points=1024)

    train_set = ModelNet10(train_points, train_labels)
    test_set = ModelNet10(test_points, test_labels)

    # Set batch size
    batch_size = 32

    # Create DataLoader instances
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=batch_size, shuffle=False)

    opt = optim.SGD(pct.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4) # optim.Adam(lr=0.0001, params=pct.parameters(
    train(pct, train_loader, cross_entropy_loss_with_label_smoothing, opt, 128)


if __name__ == '__main__':
    main()