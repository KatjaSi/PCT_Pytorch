import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from criterion import cross_entropy_loss_with_label_smoothing
import argparse
import sklearn.metrics as metrics
import numpy as np
from model import SPCT, PCT
from data_handling import parse_dataset, ModelNet

from data import load_data

def train(model:SPCT, train_loader:DataLoader, test_loader:DataLoader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model = model.double()
    model = model.to(device)
    model = nn.DataParallel(model)

    learning_rate = optimizer.param_groups[0]['lr']

    scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=learning_rate/100)

    best_test_acc = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        count = 0.0
        model.train()

        train_pred = []
        train_true = []
        idx = 0
        for data, labels in (train_loader):
            batch_size = len(labels)
            data = data.double().to(device)  # Move data to device
            labels = labels.to(device)
            data = data.permute(0, 2, 1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = outputs.max(dim=1)[1]
            count += batch_size
            running_loss += loss.item() * batch_size
            train_true.append(labels.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1
            
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Epoch: %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                    running_loss*1.0/count,
                                                                                    metrics.accuracy_score(
                                                                                    train_true, train_pred),
                                                                                    metrics.balanced_accuracy_score(
                                                                                    train_true, train_pred))
        print(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, labels in (test_loader):
            data, labels = data.double().to(device), labels.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            outputs = model(data)
            loss = criterion(outputs, labels)
            preds = outputs.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc

        outstr = 'Epoch: %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                            test_loss*1.0/count,
                                                                            test_acc,
                                                                            avg_per_class_acc)
        print(outstr)
        print(f"best test accuracy is {best_test_acc}")
    
    print(f"Finished Training, best test accuracy is {best_test_acc}")

def __parse_args__():
    parser = argparse.ArgumentParser(description='Point Cloud Classification')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch, default 32)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='Number of training epochs, default 250')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='Learning rate (0.01 by defaukt)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Num of points to sample from each point cloud')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--dataset', type=str, default="modelnet10",
                        help='dataset: modelnet10 or modelnet40')
    parser.add_argument('--model', type=str, default="PCT",
                        help='model: SPCT (Simple Point Cloud Transformer) or PCT (with neighbour embedding)')
    return parser.parse_args()

def main():
    
    args = __parse_args__()
    #train_points, test_points, train_labels, test_labels, _ = parse_dataset(num_points=args.num_points, dataset=args.dataset)
    train_points, train_labels = load_data("train")
    test_points, test_labels = load_data("test")

    train_set = ModelNet(train_points, train_labels)
    test_set = ModelNet(test_points, test_labels)
    if (args.dataset=="modelnet10"):
        train_set = ModelNet(train_points, train_labels, set_type="train")
        test_set = ModelNet(test_points, test_labels, set_type="test")
        output_channels = 10
    elif (args.dataset=="modelnet40"):
        train_set = ModelNet(train_points, train_labels, set_type="train")
        test_set = ModelNet(test_points, test_labels, set_type="test")
        output_channels = 40
    else:
        print("The dataset argument can be modelnet10 or modelnet40")
        return
    if (args.model=="SPCT"):
        pct = SPCT(output_channels=output_channels)
    elif (args.model=="PCT"):
        pct = PCT(output_channels=output_channels)
    else:
        print("The model can be SPCT or PCT")
        return

    # Set batch size
    batch_size = args.batch_size

    # Create DataLoader instances
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=batch_size, shuffle=False)

    opt = optim.SGD(pct.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4) # optim.Adam(lr=0.0001, params=pct.parameters(
    train(  model=pct,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=cross_entropy_loss_with_label_smoothing,
            optimizer=opt,
            num_epochs=args.epochs
            )

if __name__ == '__main__':
    main()