import os
import requests
import zipfile
import argparse
from tqdm import tqdm  # Import tqdm for progress bars

def download_and_unpack_zip(url, destination_folder):
    # Create the destination folder if it doesn't exist
    if os.path.exists(destination_folder):
        print(f"Folder '{destination_folder}' already exists. Skipping download and extraction.")
        return
    
    os.makedirs(destination_folder)

    # Get the filename from the URL
    filename = os.path.join(destination_folder, url.split("/")[-1])

    # Download the ZIP file
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

    # Unpack the ZIP file
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    # Remove the downloaded ZIP file if you don't need it
    os.remove(filename)

def main():
    parser = argparse.ArgumentParser(description="Download ModelNet10 or ModelNet40 dataset.")
    parser.add_argument("dataset", choices=["modelnet10", "modelnet40"], help="Specify the dataset to download")
    args = parser.parse_args()

    if args.dataset == "modelnet10":
        download_url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
        destination_folder = "./ModelNet10"
    elif args.dataset == "modelnet40":
        download_url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
        destination_folder = "./ModelNet40"
    else:
        print("Invalid dataset choice. Please choose 'modelnet10' or 'modelnet40'.")
        return

    download_and_unpack_zip(download_url, destination_folder)

if __name__ == "__main__":
    main()
