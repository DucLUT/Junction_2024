import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import model as model_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_image(image_path, transform, device):
    """
    Preprocess the image for model input.

    Args:
        image_path (str): Path to the floorplan image.
        transform (torchvision.transforms.Compose): Transformations to apply to the image.
        device (torch.device): The device to move the image to.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = (
        transform(image).unsqueeze(0).to(device)
    )  # Add batch dimension and move to device

    print("Transformed image tensor:", input_tensor)
    print(
        "Sample pixel values:", input_tensor[0, :, :5, :5]
    )  # View top-left corner pixel values

    return input_tensor


def segment_floorplan(image_path, model, transform, device, threshold=0.5):
    """
    Runs the model on a new image to extract walls (segmentation mask).

    Args:
        image_path (str): Path to the floorplan image.
        model (torch.nn.Module): The trained U-Net model.
        transform (torchvision.transforms.Compose): Transformations to apply to the image.
        device (torch.device): The device to run the model on.
        threshold (float): Threshold for converting output probabilities to binary mask.

    Returns:
        torch.Tensor: Binary mask of the segmented walls.
    """
    input_tensor = preprocess_image(image_path, transform, device)

    with torch.no_grad():
        # Get model output and apply sigmoid to get probabilities
        output = model(input_tensor)
        probabilities = torch.sigmoid(output)

        # Debug: Print max and min probabilities to ensure values make sense
        print("Max probability:", probabilities.max().item())
        print("Min probability:", probabilities.min().item())

        # Convert probabilities to binary mask using the threshold
        binary_mask = probabilities > threshold

    # Remove batch and channel dimensions, convert to 2D array
    return binary_mask.squeeze().cpu()  # Remove extra dimensions and move to CPU


def main():
    """
    Main function to load an image, run the segmentation, and save the output mask.
    """
    model_path = "model.pth"  # Path to the saved model
    image_path = "image.png"  # Replace with your image path
    output_path = "segmented_mask.png"  # Path to save the output mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize to the same size used in training
            transforms.ToTensor(),
        ]
    )

    # Run the segmentation
    binary_mask = segment_floorplan(image_path, model, transform, device, threshold=0.5)

    # Check unique values in the binary mask to confirm thresholding effect
    print("Unique values in binary mask:", np.unique(binary_mask.numpy()))

    # Convert the binary mask to a PIL image and save it
    binary_mask_np = (binary_mask.numpy() * 255).astype(np.uint8)  # Scale to 0-255
    binary_mask_image = Image.fromarray(
        binary_mask_np, mode="L"
    )  # Specify mode 'L' for grayscale
    binary_mask_image.save(output_path)
    print(f"Segmented mask saved to {output_path}")


if __name__ == "__main__":
    main()
