#https://www.vlfeat.org/api/quickshift.html

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift
from skimage.color import rgb2lab
import cv2
import numpy as np

def gaussian_kernel(distances, sigma):
    return np.exp(-0.5 * (distances / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def parzen_density_estimate(data, sigma, distance_threshold=10):
    n_points = data.shape[0]
    density = np.zeros(n_points)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(data[i] - data[j])
            if dist < distance_threshold:
                density[i] += np.exp(-0.5 * (dist / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                density[j] += np.exp(-0.5 * (dist / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return density


def quick_shift(image, sigma=1.0):
    # Convert image to suitable format for clustering
    data = image.reshape((-1, 3)).astype(np.float32)
    
    # Perform Quick Shift clustering
    parent, _, _ = quick_shift_core(data, sigma)
    clusters = assign_clusters(parent)
    
    # Reshape cluster labels to match the image shape
    segmented_image = clusters.reshape(image.shape[:2])
    
    return segmented_image

def quick_shift_core(data, sigma):
    # Calculate density estimates
    density = parzen_density_estimate(data, sigma)
    
    # Initialize parent and distance arrays
    parent = np.full(data.shape[0], -1, dtype=int)
    min_distance = np.full(data.shape[0], np.inf)
    
    # Iterate over each point to find the parent
    for i in range(data.shape[0]):
        higher_density_indices = np.where(density > density[i])[0]
        if len(higher_density_indices) > 0:
            distances = np.sum((data[higher_density_indices] - data[i])**2, axis=1)
            min_idx = np.argmin(distances)
            parent[i] = higher_density_indices[min_idx]
            min_distance[i] = distances[min_idx]
    
    return parent, min_distance, density

def assign_clusters(parent):
    # Assign cluster labels based on tree structure
    clusters = np.full(len(parent), -1, dtype=int)
    cluster_id = 0
    
    for i in range(len(parent)):
        if clusters[i] == -1:  # If not yet assigned to a cluster
            current = i
            while parent[current] != -1 and clusters[current] == -1:
                clusters[current] = cluster_id
                current = parent[current]
            cluster_id += 1
    
    return clusters

# Read the image
image = cv2.imread("images/milkshake.jpg")

# Apply Quick Shift segmentation
segmented_image = quick_shift(image)

# Save the segmented image
cv2.imwrite("segmented_image_quickshift2.jpg", segmented_image)





'''








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
    main(image_path, output_path)'''