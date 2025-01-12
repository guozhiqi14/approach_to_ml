import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from albumentations import (Compose, OneOf, RandomBrightnessContrast, RandomGamma, ShiftScaleRotate, Resize)

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Define the path to your training data
TRAIN_PATH = "../input/"


class SIIMDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_ids,
            transform=True,
            preprocessing_fn=None
    ):
        """
        Dataset class for segmentation problem
        :param image_ids: ids of the images, list
        :param transform: True/False, no transform in validation
        :param preprocessing_fn: a function for preprocessing image
        """
        # we create an empty dictionary to store image
        # and mask paths
        self.data = defaultdict(dict)

        # for augmentations
        self.transform = transform

        # preprocessing function to normalize
        # images
        self.preprocessing_fn = preprocessing_fn

        # albumentation augmentations
        # we have shift, scale & rotate
        # applied with 80% probability
        # and then one of gamma and brightness/contrast
        # is applied to the image
        # albumentation takes care of which augmentation
        # is applied to image and mask
        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.8
                ), OneOf(
                [
                    RandomGamma(
                        gamma_limit=(90, 110)),
                    RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1
                                             ), ],
                p=0.5, ),
            ])

        # Ensure image_ids is not empty
        if not image_ids:
            raise ValueError("image_ids list is empty")
        for imgid in image_ids:
            img_path = os.path.join(TRAIN_PATH, "train_png", imgid + ".png")
            mask_path = os.path.join(TRAIN_PATH, "mask_png", imgid + "_mask.png")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.data[imgid] = {
                    "img_path": img_path,
                    "mask_path": mask_path,
                }
            else:
                print(f"Warning: {img_path} or {mask_path} does not exist.")

        print(f"Total images: {len(self.data)}")  # Debugging: Print the number of images

    def __len__(self):
        # return length of dataset
        return len(self.data)

    def __getitem__(self, index):
        # for a given item index,
        # return image and mask tensors
        # read image and mask paths
        imgid = list(self.data.keys())[index]
        img_path = self.data[imgid]["img_path"]
        mask_path = self.data[imgid]["mask_path"]

        # read image and convert to RGB
        img = Image.open(img_path)
        img = img.convert("RGB")

        # PIL image to numpy array
        img = np.array(img)

        # read mask image
        mask = Image.open(mask_path)
        mask = np.array(mask)  # Convert mask to numpy array

        # Check if the dimensions of the image and mask are the same
        if img.shape[:2] != mask.shape[:2]:
            # print(f"Warning: Image and mask dimensions do not match for {imgid}. Resizing mask.")
            mask = np.array(Image.fromarray(mask).resize((img.shape[1], img.shape[0])))

        # convert to binary float matrix
        mask = (mask >= 1).astype("float32")

        # if this is training data, apply transforms
        if self.transform is True:
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # preprocess the image using provided
        # preprocessing tensors. this is basically
        # preprocess the image using provided
        # preprocessing tensors. this is basically
        # image normalization
        img = self.preprocessing_fn(img)

        # return image and mask tensors
        return {
            "image": transforms.ToTensor()(img),
            "mask": transforms.ToTensor()(mask).float(),
        }
