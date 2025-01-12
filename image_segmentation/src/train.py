import os
import sys
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from sklearn import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from dataset import SIIMDataset

# training csv file path
TRAINING_CSV = "../input/train.csv"
# training and test batch sizes
TRAINING_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4
# number of epochs
EPOCHS = 2
# define the encoder for U-Net
ENCODER = "resnet18"
# we use imagenet pretrained weights for the encoder
ENCODER_WEIGHTS = "imagenet"
# train on gpu if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataset, data_loader, model, criterion, optimizer, scaler):
    model.train()
    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)

    for d in tk0:
        inputs = d["image"]
        targets = d["mask"]
        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.float)
        optimizer.zero_grad()

        if DEVICE == "cuda":
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Batch Loss: {loss.item()}")  # Print the loss for each batch

    tk0.close()


def evaluate(dataset, data_loader, model, criterion):
    model.eval()
    final_loss = 0
    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)

    with torch.no_grad():
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]
            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)
            if DEVICE == "cuda":
                with autocast():
                    output = model(inputs)
                    loss = criterion(output, targets)
            else:
                output = model(inputs)
                loss = criterion(output, targets)
            final_loss += loss

            print(f"Validation Batch Loss: {loss.item()}")  # Print the loss for each validation batch

    tk0.close()
    return final_loss / num_batches


if __name__ == "__main__":
    print("Reading CSV file...")
    df = pd.read_csv(TRAINING_CSV)
    print(f"Total images: {len(df)}")

    print("Splitting data into training and validation sets...")
    df_train, df_valid = model_selection.train_test_split(df, random_state=42, test_size=0.1)
    training_images = df_train.ImageId.values.tolist()
    validation_images = df_valid.ImageId.values.tolist()

    print(f"Number of training images: {len(training_images)}")
    print(f"Number of validation images: {len(validation_images)}")

    # fetch unet model from segmentation models
    # with specified encoder architecture
    print("Creating model...")
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=None,
    )

    # segmentation model provides you with a preprocessing
    # function that can be used for normalizing images
    # normalization is only applied on images and not masks
    prep_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )

    model.to(DEVICE)

    print("Creating datasets and dataloaders...")
    train_dataset = SIIMDataset(
        training_images,
        transform=True,
        preprocessing_fn=prep_fn,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Reduce the number of workers if necessary
    )

    valid_dataset = SIIMDataset(
        validation_images,
        transform=False,
        preprocessing_fn=prep_fn,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0  # Reduce the number of workers if necessary
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, verbose=True
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    if torch.cuda.device_count() > 1 and DEVICE == "cuda":
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    print(f"Training batch size: {TRAINING_BATCH_SIZE}")
    print(f"Test batch size: {TEST_BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(valid_dataset)}")

    for epoch in range(EPOCHS):
        print(f"Training Epoch: {epoch}")
        train(
            train_dataset,
            train_loader,
            model,
            criterion,
            optimizer,
            scaler
        )
        print(f"Validation Epoch: {epoch}")
        val_loss = evaluate(
            valid_dataset,
            valid_loader,
            model,
            criterion
        )
        print(f"Validation Loss: {val_loss}")

        # step the scheduler
        scheduler.step(val_loss)
        print("\n")