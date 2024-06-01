import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift
from skimage.color import rgb2lab

def load_image(image_path):
    """
    Load an image from the specified path and convert it to RGB format.

    Args:
    image_path (str): The path to the image file.

    Returns:
    np.ndarray: The loaded image in RGB format.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def apply_quickshift(image, kernel_size=3, max_dist=6, ratio=0.5):
    """
    Apply Quickshift clustering to the image.

    Args:
    image (np.ndarray): The input image.
    kernel_size (int): Size of the kernel used in the quickshift algorithm.
    max_dist (int): Cut-off point for data distances.
    ratio (float): Balances color-space proximity and image-space proximity.

    Returns:
    np.ndarray: The segmented image.
    """
    image_lab = rgb2lab(image)
    segments = quickshift(image_lab, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
    return segments

def display_images(original_image, segmented_image):
    """
    Display the original and segmented images side by side.

    Args:
    original_image (np.ndarray): The original image.
    segmented_image (np.ndarray): The segmented image.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Segmented Image')
    plt.imshow(segmented_image)
    plt.axis('off')

    plt.show()

def save_image(image, output_path):
    """
    Save the image to the specified path.

    Args:
    image (np.ndarray): The image to save.
    output_path (str): The path to save the image.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)

def main(image_path, output_path, kernel_size=3, max_dist=6, ratio=0.5):
    """
    Main function to perform Quickshift image segmentation.

    Args:
    image_path (str): The path to the input image.
    output_path (str): The path to save the segmented image.
    kernel_size (int): Size of the kernel used in the quickshift algorithm.
    max_dist (int): Cut-off point for data distances.
    ratio (float): Balances color-space proximity and image-space proximity.
    """
    original_image = load_image(image_path)
    segments = apply_quickshift(original_image, kernel_size, max_dist, ratio)
    display_images(original_image, segments)
    save_image(original_image, output_path)

if __name__ == "__main__":
    image_path = 'images/milkshake.jpg'
    output_path = 'segmented_image_quickshift.jpg'
    main(image_path, output_path)