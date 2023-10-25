import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pointSSL import POINT_SSL
from data import load_data

from data_handling import  ModelNet

class PointCloudClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PointCloudClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    point_ssl = POINT_SSL()
    point_ssl.load_state_dict(torch.load('checkpoints/models/point_ssl_1000.t7'), strict=False)
    point_ssl.eval()

    #train_points, train_labels = load_data("train")


   # train_set = ModelNet(train_points, train_labels, set_type="test", num_points=2048)
   # train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=32, shuffle=True)

    point_ssl = point_ssl.half().to(device)

   # for data, labels in (train_loader):
    #    data = data.half().to(device)
     #   data = data.permute(0, 2, 1)
      #  embeddings, _ = point_ssl.__forward__(data)
       # print(embeddings.shape)
        #print(embeddings[0])
        #break

    #embeddings = point_ssl.__forward__(first_batch.to(device))
    

if __name__ == '__main__':
    main()