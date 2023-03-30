import cv2
import numpy as np

# Load the image
image = cv2.imread("image.png")

# Add Gaussian noise to the image
noisy = np.uint8(np.clip(np.random.normal(loc=0, scale=50, size=image.shape), 0, 255))
noisy_image = cv2.add(image, noisy)

# Apply non-local means filter with different window sizes
filtered1 = cv2.fastNlMeansDenoisingColored(noisy_image, None, h=50, templateWindowSize=3, searchWindowSize=7)
filtered2 = cv2.fastNlMeansDenoisingColored(noisy_image, None, h=50, templateWindowSize=7, searchWindowSize=21)
filtered3 = cv2.fastNlMeansDenoisingColored(noisy_image, None, h=50, templateWindowSize=11, searchWindowSize=33)

# Show the original and filtered images
cv2.imshow("Original", image)
cv2.imshow("Noisy", noisy_image)
cv2.imshow("Filtered 1", filtered1)
cv2.imshow("Filtered 2", filtered2)
cv2.imshow("Filtered 3", filtered3)
cv2.waitKey(0)
