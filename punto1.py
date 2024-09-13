from skimage import io, color, filters, morphology, measure, util
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = './OBJECTS.png'
image = io.imread(image_path)

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Apply a threshold to segment the image
threshold_value = filters.threshold_otsu(gray_image)
binary_image = gray_image > threshold_value

# Remove small objects using morphological operations (assuming the pens are large objects)
cleaned_image = morphology.remove_small_objects(binary_image, min_size=1000)

# Label connected components
labeled_image = measure.label(cleaned_image)

# Extract the two largest objects (the pens)
regions = measure.regionprops(labeled_image)
regions_sorted = sorted(regions, key=lambda x: x.area, reverse=True)
pens_labels = [regions_sorted[0].label, regions_sorted[1].label]

# Create a mask for the two largest objects
pens_colored_mask = np.isin(labeled_image, pens_labels)

# Create a black background image of the same size
final_colored_image = np.zeros_like(image)

# Apply the mask to the original image to keep the pens in their original color
final_colored_image[pens_colored_mask] = image[pens_colored_mask]

# Show the result
plt.imshow(final_colored_image)
plt.axis('off')
plt.show()
