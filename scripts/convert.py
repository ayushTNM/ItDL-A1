"""
convert.py

Convert list of images to npy file.

Adapted code from Usama Aleem.
https://stackoverflow.com/a/73120915/17200348
"""

from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# Replace with proper paths
data_path = "/path/to/apebase/ipfs"
save_path = "/path/to/destination"
npy_name = "cartoon64_10000.npy"

dataset = []

# Image resize and method
resolution = (64, 64)
sampling = Image.BICUBIC


def load_dataset():
    for image_name in tqdm(os.listdir(data_path)):
        try:
            image_path = os.path.join(data_path, image_name)
            image = Image.open(image_path).resize(resolution, sampling).convert("RGB")
            np_image = np.array(image)
            dataset.append(np_image)
        except Exception as e:
            print(f"Caught:\n\t{type(e).__name__}: {e}\nMake sure your dataset is correct if this keeps occurring.")


if __name__ == "__main__":
    load_dataset()
    np_dataset = np.array(dataset)
    np.save(os.path.join(save_path, npy_name), np_dataset)
