"""
Module for running inference on a trained model and visualizing the output.
"""

import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import model as model_module


def load_model(model_path, device):
    """
    Load the trained model from the specified path.

    Args:
        model_path (str): Path to the saved model.
        device (torch.device): The device to load the model on.

    Returns:
        torch.nn.Module: The loaded model.
    """
    unet_model = model_module.UNet(in_channels=3, out_channels=1).to(device)
    unet_model.load_state_dict(torch.load(model_path, map_location=device))
    unet_model.eval()
    return unet_model


def run_inference(
    trained_model,
    img_path,
    inference_device,
    img_transform,
    patch_size=(256, 256),
    stride=(128, 128),
    threshold=0.5,
):
    """
    Run inference on a sample image and visualize the output.

    Args:
        trained_model (torch.nn.Module): The trained model.
        img_path (str): Path to the sample image.
        inference_device (torch.device): The device to run the model on.
        img_transform (torchvision.transforms.Compose): Transformations to apply to the image.
        patch_size (tuple): Size of the patches to extract.
        stride (tuple): Stride for extracting patches.
        threshold (float): Threshold for converting output probabilities to binary mask.
    """
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    patch_width, patch_height = patch_size
    stride_x, stride_y = stride

    input_tensor = img_transform(image).unsqueeze(0).to(inference_device)

    binary_mask = np.zeros((height, width), dtype=np.uint8)
    draw = ImageDraw.Draw(image)

    with torch.no_grad():
        for y in range(0, height - patch_height + 1, stride_y):
            for x in range(0, width - patch_width + 1, stride_x):
                patch = image.crop((x, y, x + patch_width, y + patch_height))
                patch_tensor = img_transform(patch).unsqueeze(0).to(inference_device)
                output = trained_model(patch_tensor)
                probabilities = torch.sigmoid(output)
                binary_patch_mask = (
                    probabilities > threshold
                ).cpu().numpy().squeeze() * 255
                binary_mask[y : y + patch_height, x : x + patch_width] = (
                    binary_patch_mask
                )
                draw.rectangle(
                    [x, y, x + patch_width, y + patch_height], outline="red", width=2
                )

    binary_mask_image = Image.fromarray(binary_mask, mode="L")

    # Display the original image with patches and the binary mask
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image with Patches")
    ax[0].axis("off")
    ax[1].imshow(binary_mask_image, cmap="gray")
    ax[1].set_title("Binary Mask")
    ax[1].axis("off")
    plt.show()


# Example usage
if __name__ == "__main__":
    MODEL_PATH = "model_epoch_1_batch_520.pth"
    IMAGE_PATH = "image.png"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRANSFORM = transforms.Compose([transforms.ToTensor()])

    unet_model = load_model(MODEL_PATH, DEVICE)
    run_inference(unet_model, IMAGE_PATH, DEVICE, TRANSFORM)
