import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the Image
image = cv2.imread('milkshake.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Step 2: Preprocess the Image
# Convert the image to a 2D array of pixels
pixels = image.reshape((-1, 3))
pixels = np.float32(pixels)

# Step 3: Choose a Segmentation Method
# Define criteria, number of clusters (K) and apply K-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
K = 4  # Number of clusters
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Step 4: Apply the Segmentation Method
# Convert back the centers to 8 bit values
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]

# Reshape the image to the original image's shape
segmented_image = segmented_image.reshape(image.shape)

# Step 5: Post-process the Segmented Image (Optional)
# This step can include refining the segmentation, smoothing, etc.

# Step 6: Visualize and Save the Segmented Image
# Display the original and segmented images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()

# Save the segmented image
segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
cv2.imwrite('segmented_image.jpg', segmented_image_bgr)