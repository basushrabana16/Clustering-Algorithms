import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_shift_plus_plus(image, num_samples=1000, radius=20):
    # image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sample random points from the image
    np.random.seed(42)
    height, width = gray_image.shape
    samples = np.random.randint(0, height * width, size=(num_samples,))
    samples = np.unravel_index(samples, (height, width))
    samples = np.stack(samples, axis=-1)

    # First sample
    centers = np.array([samples[0]])

    # Centers selection using Mean Shift++
    for sample in samples[1:]:
        distances = np.linalg.norm(centers - sample, axis=1)
        if np.min(distances) >= radius:
            centers = np.vstack((centers, sample))

    # Mean Shift clustering
    centers = centers.astype(np.float32)

    # flat array of image pixels
    flat_image = gray_image.reshape((-1, 1))

    # cv2.pyrMeanShiftFiltering to each center
    segmented_image = np.zeros_like(flat_image, dtype=np.uint8)
    for center in centers:
        mask = np.linalg.norm(flat_image - flat_image[center[0] * width + center[1]], axis=1) < radius
        segmented_image[mask] = flat_image[center[0] * width + center[1]]

    # Reshape the segmented image to the original shape
    segmented_image = segmented_image.reshape(gray_image.shape)

    return segmented_image

# Step 1: Loading the Image
image = cv2.imread('images/milkshake.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Step 3: Applying the Segmentation Method
segmented_image = mean_shift_plus_plus(image)

# Step 5: Visualize and Save the Segmented Image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image (Mean Shift++)')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.show()

# Save the segmented image
cv2.imwrite('segmented_image_mean_shift_pp.jpg', segmented_image)