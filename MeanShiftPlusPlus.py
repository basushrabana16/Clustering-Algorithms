import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_shift_plus_plus(image, num_samples=1000, radius=20):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sample random points from the image
    np.random.seed(42)  # For reproducibility
    height, width = gray_image.shape
    samples = np.random.randint(0, height * width, size=(num_samples,))
    samples = np.unravel_index(samples, (height, width))
    samples = np.stack(samples, axis=-1)

    # Initialize with the first sample
    centers = np.array([samples[0]])

    # Iteratively select centers using Mean Shift++
    for sample in samples[1:]:
        distances = np.linalg.norm(centers - sample, axis=1)
        if np.min(distances) >= radius:
            centers = np.vstack((centers, sample))

    # Convert the centers to 8-bit values
    centers = np.uint8(centers)

    # Perform Mean Shift clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels = cv2.meanShift(gray_image, None, criteria)

    # Create the segmented image
    segmented_image = centers[labels.flatten()]

    # Reshape the segmented image to the original shape
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image



# Step 1: Loading the Image
image = cv2.imread('images/milkshake.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Step 2: Preprocess the Image
# No preprocessing required for Mean Shift++ clustering

# Step 3: Applying the Segmentation Method
segmented_image = mean_shift_plus_plus(image)

# Step 4: Post-process the Segmented Image (Optional)
# This step can include refining the segmentation, smoothing, etc.

# Step 5: Visualize and Save the Segmented Image
# Display the original and segmented images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image (Mean Shift++)')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()

# Save the segmented image
segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
cv2.imwrite('segmented_image_mean_shift_pp.jpg', segmented_image_bgr)
