import os
import urllib.request

def download_tsp_files():
    files_to_download = [
        'eil51', 'berlin52', 'eil76', 'kroA100', 'kroB100',
        'ch130', 'ch150', 'kroA200', 'tsp225', 'a280'
    ]
    
    base_url = "https://raw.githubusercontent.com/mastqe/tsplib/master/"
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    for name in files_to_download:
        tsp_url = f"{base_url}{name}.tsp"
        tsp_path = os.path.join(data_dir, f"{name}.tsp")
        
        print(f"Downloading {name}...")
        try:
            # We overwrite existing mock files
            urllib.request.urlretrieve(tsp_url, tsp_path)
            print(f"Successfully downloaded {name}.tsp")
        except Exception as e:
            print(f"Failed to download {name}: {e}")

if __name__ == "__main__":
    download_tsp_files()
