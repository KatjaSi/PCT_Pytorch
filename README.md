# Point Cloud Transformer

The original code is from https://github.com/Strawberry-Eat-Mango/PCT_Pytorch and https://github.com/MenghaoGuo/PCT

## Usage

### To download the dataset, follow these steps:

1. Open a terminal or command prompt.

2. Navigate to the directory containing the script downloadmodelnet.py.

3. Run the script with the following command, specifying the dataset you want to download:

    - `python download_modelnet.py modelnet10`

Replace modelnet10 with modelnet40 if you want to download the ModelNet40 dataset.

The script will download and extract the dataset to a folder named ModelNet10 or ModelNet40 in the same directory as the script.

### Training the PCT model

1. `python train.py --help` to see all the options available.

2. Example of training the model: 
    `python train.py --batch_size 64 --dataset "modelnet40" --epochs 250 --model=SPCT_FA --num_points 2048`