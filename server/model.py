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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FloorplanSegmentationDataset(Dataset):
    """
    Dataset class for floorplan segmentation.
    """

    def __init__(
        self,
        floorplan_dir,
        annotation_file,
        transform=None,
        patch_size=(128, 128),  # Reduce the patch size
        stride=(64, 64),  # Adjust the stride to balance patch overlap
        limit=None,
    ):
        """
        Args:
            floorplan_dir (str): Path to the directory containing floorplan images.
            annotation_file (str): Path to the JSON file containing annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            patch_size (tuple): The desired output patch size.
            stride (tuple): The stride for sliding window.
            limit (int, optional): Maximum number of images to load for testing.
        """
        self.floorplan_dir = floorplan_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
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
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.floorplan_dir, img_name)

        # Log image loading
        logger.info(f"Loading image {img_name} from {img_path}")
        image = Image.open(img_path).convert("RGB")
        logger.info(f"Loaded image size: {image.size}")

        # Get the corresponding annotation for this image
        img_id = os.path.splitext(img_name)[0]
        mask = np.zeros((image.height, image.width), dtype=np.float32)

        # Fill the mask with wall annotations
        for data in self.annotations["rect"].values():
            if str(data["floorID"]) == img_id:
                x_coords = np.array(data["x"], dtype=int)
                y_coords = np.array(data["y"], dtype=int)
                mask[y_coords, x_coords] = 1  # Mark these as walls

        mask = Image.fromarray(mask)

        # Extract a single patch
        img_patch, mask_patch = self.extract_patch(image, mask)

        # Log patch extraction details
        logger.info(f"Extracted patch size: {img_patch.size}")

        # Apply transformations if any
        if self.transform:
            img_patch = self.transform(img_patch)
            mask_patch = torch.from_numpy(np.array(mask_patch)).float().unsqueeze(0)

        # Return as tensors (ensure that it returns an image and mask tuple)
        return img_patch, mask_patch

    def extract_patch(self, image: Image.Image, mask: Image.Image):
        """
        Extract a single patch from the image and mask.

        Args:
            image (Image.Image): The input image.
            mask (Image.Image): The corresponding mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A single image and mask patch.
        """
        img_width, img_height = image.size
        patch_width, patch_height = self.patch_size
        stride_x, stride_y = self.stride

        # Randomly select a patch position to avoid bias from the dataset
        x = np.random.randint(0, img_width - patch_width)
        y = np.random.randint(0, img_height - patch_height)

        img_patch = image.crop((x, y, x + patch_width, y + patch_height))
        mask_patch = mask.crop((x, y, x + patch_width, y + patch_height))

        return img_patch, mask_patch


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
            images, masks = images.cuda(), masks.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Ensure the masks have a channel dimension
            masks = masks.unsqueeze(1)  # Add channel dimension to masks if needed
            masks = masks.squeeze(1)  # Remove channel dimension if needed

            # Compute loss
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Log GPU memory usage
            if torch.cuda.is_available():
                logger.info(
                    f"GPU Memory Allocated (after step {batch_idx}): {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
                )
                logger.info(
                    f"GPU Memory Cached (after step {batch_idx}): {torch.cuda.memory_reserved() / 1024**2:.2f} MB"
                )

            # Update the progress bar
            epoch_loss += loss.item()
            loop.set_postfix(loss=epoch_loss / (batch_idx + 1))

        logger.info("Epoch %d - Loss: %.4f", epoch + 1, epoch_loss / num_batches)

    return model


def evaluate_model(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_idx, batch in enumerate(val_loader):
            logger.info("Processing batch %d", batch_idx)

            # Extract images and masks from the batch
            images = batch[0]  # The first element is the image tensor
            masks = batch[1]  # The second element is the mask tensor

            # Move tensors to the correct device
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)

            # Log or print evaluation results (loss, etc.)

            logger.info("Loss: %.4f", loss.item())


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
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

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
    logger.info(f"Trained model saved at {save_path}")

    logger.info("Starting evaluation...")
    evaluate_model(trained_model, val_loader, device)
    logger.info("Evaluation completed.")


if __name__ == "__main__":
    main()
