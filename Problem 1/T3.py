import cv2
import numpy as np

# Load the image
img = cv2.imread('image.png')

# Add Gaussian noise to the image
mean = 0
var = 100
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, img.shape)
noisy_image = np.clip(img + gaussian, 0, 255).astype(np.uint8)

# Apply Wiener filter
filtered_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
filtered_image = cv2.medianBlur(filtered_image, 3)
filtered_image = cv2.fastNlMeansDenoising(filtered_image, None, h=10, searchWindowSize=21)
filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)

# Show the original and filtered images
cv2.imshow('Original', img)
cv2.imshow('Noisy', noisy_image)
cv2.imshow('Filtered', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
