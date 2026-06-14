import os
import urllib.request

def download_file(url, destination):
    # Ensure destination folder exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(f"Downloading from {url}...")
    try:
        # User-Agent header is set to prevent HTTP 403 Forbidden errors
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req) as response:
            with open(destination, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Successfully downloaded to {destination}!")
        print(f"File size: {os.path.getsize(destination)} bytes")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise e

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/prernagoel/IBM-Employee-Attrition-Analysis-in-Python/master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    
    # We define absolute paths or paths relative to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(current_dir, "Dataset", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    download_file(url, dest)
