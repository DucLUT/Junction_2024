import os
import json
import logging
from typing import List, Tuple

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import tqdm  # For progress bar
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class FloorplanSegmentationDataset(Dataset):
    """
    Dataset class for floorplan segmentation with lazy loading and batching of patches.
    """

    def __init__(
        self,
        floorplan_dir,
        annotation_file,
        transform=None,
        patch_size=(128, 128),
        stride=(64, 64),
        limit=None,
        patches_per_batch=10,  # Number of patches to load in each batch per image
    ):
        """
        Args:
            floorplan_dir (str): Path to the directory containing floorplan images.
            annotation_file (str): Path to the JSON file containing annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            patch_size (tuple): The desired output patch size.
            stride (tuple): The stride for sliding window.
            limit (int, optional): Maximum number of images to load for testing.
            patches_per_batch (int): Number of patches to load per batch per image.
        """
        self.floorplan_dir = floorplan_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.patches_per_batch = patches_per_batch
        self.counter = 0

        logger.info("Loading annotations from %s", annotation_file)
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        logger.info("Listing all floorplan images in %s", floorplan_dir)
        self.image_files = [f for f in os.listdir(floorplan_dir) if f.endswith(".png")]

        # Apply the limit if specified
        if limit is not None:
            self.image_files = self.image_files[:limit]

        logger.info("Found %d floorplan images", len(self.image_files))

    def __len__(self):
        total_patches = 0
        for img_file in self.image_files:
            img_path = os.path.join(self.floorplan_dir, img_file)
            image = Image.open(img_path).convert("RGB")
            mask = self.get_mask(image, img_file)
            img_patches, _ = self.extract_all_patches(image, mask, img_file)
            total_patches += len(img_patches)

        # Log total number of patches
        logger.info("Total patches in the dataset: %d", total_patches)

        return total_patches // self.patches_per_batch

    def __getitem__(self, idx):
        """
        Get a batch of patches from the dataset.
        Instead of returning a single image, return a batch of patches.
        """
        img_batch = []
        mask_batch = []

        total_patches = 0
        for img_name in self.image_files:
            img_path = os.path.join(self.floorplan_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            mask = self.get_mask(image, img_name)

            # Extract all patches from this image
            img_patches, mask_patches = self.extract_all_patches(image, mask, img_name)

            num_patches = len(img_patches)
            if total_patches + num_patches > idx * self.patches_per_batch:
                start_patch = (idx * self.patches_per_batch) - total_patches
                end_patch = start_patch + self.patches_per_batch

                img_batch.extend(img_patches[start_patch:end_patch])
                mask_batch.extend(mask_patches[start_patch:end_patch])
                break

            total_patches += num_patches

        # Log progress of loading patches
        logger.info("Loaded %d patches for batch %d", len(img_batch), idx)

        if self.transform:
            img_batch = [self.transform(patch) for patch in img_batch]
            mask_batch = [
                torch.from_numpy(np.array(mask_patch)).float().unsqueeze(0)
                for mask_patch in mask_batch
            ]

        # Stack the patches into a 4D tensor
        img_batch = torch.stack(img_batch)
        mask_batch = torch.stack(mask_batch)

        return img_batch, mask_batch

    def get_mask(self, image: Image.Image, filename: str):
        """
        Get the mask for a given image based on the annotations.

        Args:
            image (Image.Image): The input image.
            filename (str): The image file name used to get the correct mask.

        Returns:
            Image.Image: The mask image for the floorplan.
        """
        img_id = os.path.splitext(filename)[0]  # Extract filename without extension
        mask = np.zeros((image.height, image.width), dtype=np.float32)

        # Fill the mask with wall annotations
        for data in self.annotations["rect"].values():
            if str(data["floorID"]) == img_id:
                x_coords = np.array(data["x"], dtype=int)
                y_coords = np.array(data["y"], dtype=int)
                mask[y_coords, x_coords] = 1  # Mark these as walls

        return Image.fromarray(mask)

    def extract_all_patches(self, image: Image.Image, mask: Image.Image, filename: str):
        """
        Extract all patches from the image and mask using sliding window.
        """
        img_width, img_height = image.size
        patch_width, patch_height = self.patch_size
        stride_x, stride_y = self.stride

        img_patches = []
        mask_patches = []

        # Slide a window across the image to extract all patches
        for y in range(0, img_height - patch_height + 1, stride_y):
            for x in range(0, img_width - patch_width + 1, stride_x):
                img_patch = image.crop((x, y, x + patch_width, y + patch_height))
                mask_patch = mask.crop((x, y, x + patch_width, y + patch_height))

                img_patches.append(img_patch)
                mask_patches.append(mask_patch)

        # Log the number of patches extracted
        logger.info("Extracted %d patches from image %s.", len(img_patches), filename)
        return img_patches, mask_patches


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

        # Add an upsampling layer to match target size
        self.upsample = nn.Upsample(
            size=(256, 256), mode="bilinear", align_corners=False
        )

    def forward(self, x):
        """
        Forward pass of the U-Net model.
        """
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        x3 = self.upsample(x3)  # Ensure output size is 256x256
        return x3

    def summary(self):
        """
        Print the summary of the model.
        """
        logger.info("Model Summary:")
        logger.info(self)


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Train the U-Net model with logging for memory usage and loss tracking.

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

        for batch_idx, batch in enumerate(loop):
            images, masks = batch
            # Reshape images and masks to merge patches per batch into batch dimension
            batch_size, patches_per_batch, channels, height, width = images.shape
            images = images.view(
                batch_size * patches_per_batch, channels, height, width
            )
            masks = masks.view(batch_size * patches_per_batch, 1, height, width)

            images, masks = images.cuda(), masks.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Log GPU memory usage and loss
            if torch.cuda.is_available():
                logger.info(
                    "Epoch %d, Batch %d/%d - GPU Memory Allocated: %.2f MB",
                    epoch + 1,
                    batch_idx + 1,
                    num_batches,
                    torch.cuda.memory_allocated() / 1024**2,
                )
                logger.info(
                    "Epoch %d, Batch %d/%d - GPU Memory Cached: %.2f MB",
                    epoch + 1,
                    batch_idx + 1,
                    num_batches,
                    torch.cuda.memory_reserved() / 1024**2,
                )

            # Update the progress bar
            epoch_loss += loss.item()
            loop.set_postfix(loss=epoch_loss / (batch_idx + 1))

            # Save model after every 20 batches
            if (batch_idx + 1) % 20 == 0:
                batch_model_path = os.path.join(
                    ".", f"model_epoch_{epoch+1}_batch_{batch_idx+1}.pth"
                )
                torch.save(model.state_dict(), batch_model_path)
                logger.info("Saved model to %s", batch_model_path)

        logger.info("Epoch %d - Loss: %.4f", epoch + 1, epoch_loss / num_batches)

        # Save model after every epoch
        model_path = os.path.join(".", f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info("Saved model to %s", model_path)

    return model


def evaluate_model(model, val_loader, device):
    """
    Evaluate the U-Net model on the validation data.

    Args:
        model (nn.Module): The U-Net model.
        val_loader (DataLoader): DataLoader for the validation data.
        device (torch.device): The device (CPU or CUDA) on which the model and data are loaded.
    """
    model.eval()  # Set the model to evaluation mode
    total_dice_score = 0.0
    num_samples = 0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for batch_idx, batch in enumerate(val_loader):
            logger.info("Processing batch %d", batch_idx + 1)

            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)

            # Convert logits to binary predictions (0 or 1) using a threshold of 0.5
            preds = torch.sigmoid(outputs) > 0.5

            # Ensure preds and masks are float for Dice score computation
            preds = preds.float()
            masks = masks.float()

            # Calculate intersection and union for Dice score
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum()

            # Avoid division by zero by checking if the intersection and union are both zero
            if union + intersection > 0:
                dice_score = 2 * intersection / (union + intersection)
            else:
                dice_score = torch.tensor(
                    1.0
                )  # If both are empty, assume perfect match

            total_dice_score += dice_score.item()
            num_samples += 1

            logger.info("Batch %d - Dice Score: %.4f", batch_idx + 1, dice_score.item())

    # Calculate average Dice score
    avg_dice_score = total_dice_score / num_samples if num_samples > 0 else 0.0
    logger.info("Average Dice Score: %.4f", avg_dice_score)


def main():
    """
    Main script for data preparation, model training, and evaluation.
    """
    floorplan_dir = "MLSTRUCT-FP_v1"
    annotation_file = "MLSTRUCT-FP_v1/fp.json"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Set a limit of 10 images for faster testing
    dataset = FloorplanSegmentationDataset(
        floorplan_dir,
        annotation_file,
        transform=transform,
        patch_size=(256, 256),
        stride=(128, 128),
        limit=10,
    )

    # Split dataset into training and validation sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=2, sampler=train_sampler, num_workers=2
    )
    val_loader = DataLoader(dataset, batch_size=2, sampler=val_sampler, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    logger.info("Starting training...")
    trained_model = train_model(
        model, train_loader, criterion, optimizer, num_epochs=10
    )
    logger.info("Training completed.")

    # Save the trained model
    save_path = "model.pth"
    torch.save(trained_model.state_dict(), save_path)
    logger.info("Trained model saved at %s", save_path)

    logger.info("Starting evaluation...")
    evaluate_model(trained_model, val_loader, device)
    logger.info("Evaluation completed.")


if __name__ == "__main__":
    main()
