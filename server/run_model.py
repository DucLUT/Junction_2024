import logging
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import model as model_module
import os
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FloorplanSegmentationDataset(Dataset):
    """
    Dataset class for segmenting floorplan images into patches.
    """

    def __init__(
        self, image_path, transform=None, patch_size=(256, 256), stride=(128, 128)
    ):
        """
        Initialize the dataset with image path, transform, patch size, and stride.

        Args:
            image_path (str): Path to the floorplan image.
            transform (callable, optional): Optional transform to be applied on a patch.
            patch_size (tuple, optional): Size of the patches to extract.
            stride (tuple, optional): Stride for extracting patches.
        """
        self.image = Image.open(image_path).convert("RGB")
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.patches = self.extract_patches()

    def extract_patches(self):
        """
        Extract patches from the image.

        Returns:
            list: List of image patches.
        """
        patches = []
        width, height = self.image.size
        patch_width, patch_height = self.patch_size
        stride_x, stride_y = self.stride

        for y in range(0, height - patch_height + 1, stride_y):
            for x in range(0, width - patch_width + 1, stride_x):
                patch = self.image.crop((x, y, x + patch_width, y + patch_height))
                print(
                    f"Extracting patch at position: ({x}, {y})"
                )  # Add this line to log positions
                if self.transform:
                    patch = self.transform(patch)
                patches.append(patch)

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]


def load_model(model_path, device):
    """
    Load the trained model from the specified path.

    Args:
        model_path (str): Path to the saved model.
        device (torch.device): The device to load the model on.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = model_module.UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def segment_floorplan_patches(data_loader, model, device, threshold=0.5):
    """
    Runs the model on patches of images to extract walls (segmentation mask).

    Args:
        data_loader (DataLoader): DataLoader for the floorplan patches.
        model (torch.nn.Module): The trained U-Net model.
        device (torch.device): The device to run the model on.
        threshold (float): Threshold for converting output probabilities to binary mask.

    Returns:
        list: List of binary masks for each patch.
    """
    binary_masks = []

    with torch.no_grad():
        for img_batch in data_loader:
            img_batch = img_batch.to(device)
            output = model(img_batch)
            probabilities = torch.sigmoid(output)
            binary_mask = probabilities > threshold
            binary_masks.append(binary_mask.cpu())

    return binary_masks


def main():
    """
    Main function to load an image, run the segmentation, and save the output mask.
    """
    model_path = "model_epoch_1_batch_460.pth"
    image_path = "image.png"
    output_dir = "output_masks"

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    array = cv2.imread(image_path)
    if np.any(array):
        print(f"Image loaded from {image_path}")
    else:
        print(f"Failed to load image from {image_path}")

    dataset = FloorplanSegmentationDataset(
        image_path, transform=transform, patch_size=(256, 256), stride=(128, 128)
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    binary_masks = segment_floorplan_patches(data_loader, model, device, threshold=0.5)

    for i, binary_mask in enumerate(binary_masks):
        binary_mask_np = (binary_mask.numpy() * 255).astype(np.uint8)
        binary_mask_image = Image.fromarray(binary_mask_np.squeeze(), mode="L")
        output_path = f"{output_dir}/segmented_mask_{i}.png"
        if np.any(binary_mask_np):
            binary_mask_image.save(output_path)
            print(f"Segmented mask saved to {output_path}")
        else:
            continue


if __name__ == "__main__":
    main()
