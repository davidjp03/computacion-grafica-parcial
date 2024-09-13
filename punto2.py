from skimage import io, img_as_float, filters, restoration, color, morphology, segmentation, feature
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = './danger.jpg'
image = io.imread(image_path)

# Convert to grayscale for processing
gray_image = color.rgb2gray(image)

# Apply Gaussian filtering to reduce noise
gaussian_filtered = filters.gaussian(gray_image, sigma=1)

# Edge detection using Sobel filter
edges_sobel = filters.sobel(gaussian_filtered)

# Thresholding for segmentation
thresh = filters.threshold_otsu(gaussian_filtered)
segmentation_result = gaussian_filtered > thresh

# Apply morphological operations (dilation followed by erosion)
dilated = morphology.dilation(segmentation_result, morphology.square(3))
eroded = morphology.erosion(dilated, morphology.square(3))

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(gaussian_filtered, cmap='gray')
axes[0, 1].set_title('Gaussian Filtered')
axes[0, 1].axis('off')

axes[0, 2].imshow(edges_sobel, cmap='gray')
axes[0, 2].set_title('Edge Detection (Sobel)')
axes[0, 2].axis('off')

axes[1, 0].imshow(segmentation_result, cmap='gray')
axes[1, 0].set_title('Segmentation Result')
axes[1, 0].axis('off')

axes[1, 1].imshow(dilated, cmap='gray')
axes[1, 1].set_title('Morphology (Dilated)')
axes[1, 1].axis('off')

axes[1, 2].imshow(eroded, cmap='gray')
axes[1, 2].set_title('Morphology (Eroded)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
