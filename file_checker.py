import numpy as np
import glob
import os

# Folder with .npz files
folder = "./2"   # change this to your folder path

# Find all .npz files in folder
files = glob.glob(os.path.join(folder, "*.npz"))

for file in files:
    print(f"\n=== Reading {file} ===")
    data = np.load(file, allow_pickle=True)

    # Show keys
    print("Keys:", data.files)

    # Iterate over all arrays inside
    for key in data.files:
        print(f"{key}:", data[key])
    break