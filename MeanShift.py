import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

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

def preprocess_image(image):
    """
    Preprocess the image by reshaping it to a 2D array of pixels.

    Args:
    image (np.ndarray): The input image.

    Returns:
    np.ndarray: The reshaped image as a 2D array of pixels.
    """
    pixels = image.reshape((-1, 3))
    return np.float32(pixels)

def apply_meanshift(pixels):
    """
    Apply mean shift clustering to the image pixels.

    Args:
    pixels (np.ndarray): The input pixels.

    Returns:
    tuple: The labels and cluster centers from mean shift clustering.
    """
    bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(pixels)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    return labels, cluster_centers

def segment_image(labels, centers, image_shape):
    """
    Create a segmented image from the mean shift clustering results.

    Args:
    labels (np.ndarray): The labels from mean shift clustering.
    centers (np.ndarray): The cluster centers from mean shift clustering.
    image_shape (tuple): The shape of the original image.

    Returns:
    np.ndarray: The segmented image.
    """
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    return segmented_image.reshape(image_shape)

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

def main(image_path, output_path):
    """
    Main function to perform mean shift image segmentation.

    Args:
    image_path (str): The path to the input image.
    output_path (str): The path to save the segmented image.
    """
    original_image = load_image(image_path)
    pixels = preprocess_image(original_image)
    labels, centers = apply_meanshift(pixels)
    segmented_image = segment_image(labels, centers, original_image.shape)
    display_images(original_image, segmented_image)
    save_image(segmented_image, output_path)

if __name__ == "__main__":
    image_path = 'images/milkshake.jpg'
    output_path = 'segmented_image_meanshift.jpg'
    main(image_path, output_path)
