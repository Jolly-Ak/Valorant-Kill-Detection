
import os 

def image_count (path):
    # List all files in the directory
    files = os.listdir(path)
    # Count the number of files
    count = len(files)
    return count
