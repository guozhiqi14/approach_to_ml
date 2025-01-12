import pandas as pd
import numpy as np
import os
from PIL import Image

# Load the CSV file
train_df = pd.read_csv('../input/train.csv')


def rle_decode(mask_rle, shape):
    """Decode run-length encoding to binary mask."""
    s = np.fromstring(mask_rle, sep=' ', dtype=np.uint32)
    starts, lengths = s[::2], s[1::2]
    starts -= 1  # Convert to 0-based indexing
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    return img.reshape(shape).T  # Reshape to original dimensions and transpose


# Define the output directory for masks
mask_dir = '../input/mask_png'
os.makedirs(mask_dir, exist_ok=True)


# Determine image dimensions (example)
# You can also load an actual image from your dataset if needed
height = 256  # Set based on your dataset
width = 256   # Set based on your dataset


# Loop through each row in the DataFrame
for index, row in train_df.iterrows():
    img_id = row['ImageId']  # Assuming this is the column name for image IDs
    rle_mask = row['EncodedPixels']  # Assuming this is the column name for RLE masks

    if pd.notna(rle_mask):  # Check if there is a mask for this image
        # Decode RLE to binary mask
        mask = rle_decode(rle_mask, shape=(height, width))  # Specify height and width of the images

        # Save the mask as a PNG file
        mask_path = os.path.join(mask_dir, f"{img_id}_mask.png")
        Image.fromarray(mask * 255).save(mask_path)  # Convert binary mask to uint8 and save