import os
import requests
import zipfile

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
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    # Unpack the ZIP file
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    # Remove the downloaded ZIP file if you don't need it
    os.remove(filename)

def main():
    download_url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    destination_folder = "./ModelNet10"  # Change this to your desired destination folder
    download_and_unpack_zip(download_url, destination_folder)

if __name__ == "__main__":
    main()
