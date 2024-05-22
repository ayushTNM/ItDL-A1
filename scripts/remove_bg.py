"""
remove_bg.py

Attempts to remove background from already converted Apebase npy file. Mileage may vary.
"""

import os
import numpy as np
from tqdm import tqdm

# Replace with proper paths
data_path = "/path/to/apebase/ipfs"
save_path = "/path/to/destination"
data_name = "apebase64_10000.npy"
save_name = "apebase64_10000_nobg.npy"

# Tolerance for background removal for ape edges
tol = 10


def convert():
    dataset = np.load(os.path.join(data_path, data_name))
    
    for image in tqdm(dataset):
        # Remove background
        # Sampling pixel (3, 3) (assuming 64x size) to make sure it's part of the homogenous background color
        # Few samples might have an issue where subject colors are similar to the background,
        # however this is a minority and it only makes those elements white; outlines are retained.
        mask = np.where(np.sum(np.abs(np_image - np_image[3, 3]), axis=2) < tol, True, False)
        
        # Remove corner artifacts
        # mask[:2, :2], mask[-2:, :2], mask[:2, -2:], mask[-2:, -2:] = [True]*4
        
        # Remove edge artifacts
        mask[:, :2], mask[:, -2:], mask[:2, :], mask[-2:, :] = [True]*4
        
        image[mask] = 255
    
    np.save(os.path.join(save_path, save_name), dataset)

if __name__ == "__main__":
    convert()