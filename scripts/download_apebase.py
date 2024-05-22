"""
download_apebase.py

Downloader function to convert Apebase ipfs files to npy file.

Adapted code from Usama Aleem and jarmod.
https://stackoverflow.com/a/73120915/17200348
https://stackoverflow.com/a/68941273/17200348
"""

import urllib.request
from retry import retry    
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


link = "https://ipfs.io/ipfs/"

# Replace with proper paths
data_path = "/path/to/apebase/ipfs"
save_path = "/path/to/destination"
npy_name = "apebase64_10000_nobg.npy"

# Whether or not to immediately remove the background on conversion
no_bg = True

dataset = []

# Image resize and method
resolution = (64, 64)
sampling = Image.BICUBIC

# Tolerance for background removal for ape edges
tol = 5

@retry(urllib.error.URLError, tries=4)
def load_dataset():
    for image_name in tqdm(os.listdir(data_path)):
        try:
            image_path = os.path.join(link, image_name)
            image, _ = urllib.request.urlretrieve(image_path)
            pil_image = Image.open(image).convert("RGB")
            np_image = np.array(pil_image)
            
            if no_bg:
                # Remove background
                # Sampling pixel (15, 15) to make sure it's part of the homogenous background color
                # Few samples might have an issue where subject colors are similar to the background,
                # however this is a minority and it only makes those elements white; outlines are retained.
                mask = np.where(np.sum(np.abs(np_image - np_image[15, 15]), axis=2) < tol, True, False)
                
                # Remove corner artifacts
                # mask[:2, :2], mask[-2:, :2], mask[:2, -2:], mask[-2:, -2:] = [True]*4
                
                # Remove edge artifacts
                # Margin of 8 might be too large, but guarantees no artifacts remain after downscaling
                mask[:, :8], mask[:, -8:], mask[:8, :], mask[-8:, :] = [True]*4
                
                np_image[mask] = 255
            
            dataset.append(np.array(Image.fromarray(np_image).resize(resolution, sampling)))
        except Exception as e:
            # This may also happen on HTTP errors. Some Apebase images seem to have vanished.
            print(f"Caught:\n\t{type(e).__name__}: {e}\nMake sure your dataset is correct if this keeps occurring.")
            print(f"Image missed: {image_name}")


if __name__ == "__main__":
    load_dataset()
    np_dataset = np.array(dataset)
    np.save(os.path.join(save_path, npy_name), np_dataset)
