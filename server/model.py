"""
This module provides a dataset class for floorplans, a U-Net model for segmentation,
and functions for training and evaluating the model.
"""

import os
import json
import logging
from typing import List

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import tqdm  # For progress bar
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FloorplanSegmentationDataset(Dataset):
    """
    Dataset class for floorplan segmentation.
    """

    def __init__(
        self, floorplan_dir, annotation_file, transform=None, img_size=(256, 256)
    ):
        """
        Args:
            floorplan_dir (str): Path to the directory containing floorplan images.
            annotation_file (str): Path to the JSON file containing annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            img_size (tuple): The desired output image and mask size.
        """
        self.floorplan_dir = floorplan_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.img_size = img_size
        self.counter = 0

        logger.info("Loading annotations from %s", annotation_file)
        # Load the annotations
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        logger.info("Listing all floorplan images in %s", floorplan_dir)
        # List all floorplan images (assuming they are PNGs)
        self.image_files = [f for f in os.listdir(floorplan_dir) if f.endswith(".png")]

        logger.info("Found %d floorplan images", len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.floorplan_dir, img_name)

        # Load image and resize
        image = (
            Image.open(img_path)
            .convert("RGB")
            .resize(self.img_size, Image.Resampling.LANCZOS)
        )

        # Get the corresponding annotation for this image
        img_id = os.path.splitext(img_name)[0]
        mask = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.float32)

        # Fill the mask with wall annotations
        for data in self.annotations["rect"].values():
            if str(data["floorID"]) == img_id:
                x_coords = np.array(data["x"], dtype=int)
                y_coords = np.array(data["y"], dtype=int)
                mask[y_coords, x_coords] = 1  # Mark these as walls

        mask = Image.fromarray(mask).resize(self.img_size, Image.Resampling.NEAREST)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)  # Convert mask to tensor

        logger.info(
            "Loaded %d/%d images (%.2f%%)",
            self.counter + 1,
            len(self.image_files),
            (self.counter + 1) / len(self.image_files) * 100,
        )
        self.counter += 1

        return image, mask


class UNet(nn.Module):
    """
    U-Net model for segmentation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        """
        Forward pass of the U-Net model.
        """
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

    def summary(self):
        """
        Print the summary of the model.
        """
        logger.info("Model Summary:")
        logger.info(self)


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Train the U-Net model.

    Args:
        model (nn.Module): The U-Net model.
        train_loader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        nn.Module: The trained model.
    """
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = len(train_loader)

        # Progress bar
        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

        for images, masks in loop:
            images, masks = images.cuda(), masks.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Ensure the masks have a channel dimension
            masks = masks.unsqueeze(1)  # Add channel dimension to masks if needed

            # Compute loss
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the progress bar
            epoch_loss += loss.item()
            loop.set_postfix(loss=epoch_loss / (loop.n + 1))

        logger.info("Epoch %d - Loss: %.4f", epoch + 1, epoch_loss / num_batches)

    return model


def evaluate_model(model, val_loader):
    """
    Evaluate the U-Net model.

    Args:
        model (nn.Module): The U-Net model.
        val_loader (DataLoader): DataLoader for the validation data.
    """
    model.eval()  # Set the model to evaluation mode
    total_dice_score = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.cuda(), masks.cuda()

            # Forward pass
            outputs = model(images)

            # Convert logits to binary (0 or 1)
            preds = torch.sigmoid(outputs) > 0.5

            # Calculate Dice score
            intersection = (preds & masks).sum()
            union = preds.sum() + masks.sum()
            dice_score = 2 * intersection / (union + intersection)

            total_dice_score += dice_score.item()
            num_samples += 1

    avg_dice_score = total_dice_score / num_samples
    logger.info("Average Dice Score: %.4f", avg_dice_score)


def main():
    """
    Main script for data preparation, model training, and evaluation.
    """
    floorplan_dir = "MLSTRUCT-FP_v1"
    annotation_file = "MLSTRUCT-FP_v1/fp.json"

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Ensure all images and masks are (256, 256)
            transforms.ToTensor(),
        ]
    )

    dataset = FloorplanSegmentationDataset(
        floorplan_dir, annotation_file, transform=transform, img_size=(256, 256)
    )
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=4)

    model = UNet(in_channels=3, out_channels=1).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    logger.info("Starting training...")
    trained_model = train_model(
        model, train_loader, criterion, optimizer, num_epochs=10
    )
    logger.info("Training completed.")

    logger.info("Starting evaluation...")
    evaluate_model(trained_model, val_loader)
    logger.info("Evaluation completed.")


if __name__ == "__main__":
    main()
