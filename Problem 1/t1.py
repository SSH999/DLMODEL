import cv2
import numpy as np
w=int(input("width"))
h=int(input("height"))

# Load image
img = cv2.imread('image.png')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load template
template = cv2.imread('image.png', 0)

# Apply template matching
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

# Set threshold and find location of the detected object
threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# Display result
cv2.imshow('Detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
