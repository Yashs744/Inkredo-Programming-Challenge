# Libraries
import cv2
import imutils
import numpy as np

# Path to the Image
image_path = "../DataSet/DIQ_Part1/set12/2012-04-19_18-27-07_716.jpg"

# Steps to Detect Document in the Image
#     1. Load the Image and resize it.
#     2. Convert the Image to GrayScale.
#     3. Use Bilateral Filter to Smooth out the Image.
#     4. Detect Edges using Canny Edge Detector.
#     5. Find Contours in the Edge Detected Image.
#     6. Pick Points which Cover the Document.

# 1. Read the Image using opencv imread() function
image = cv2.imread(image_path)

# 2. Resize the Image to make it easier to work with
height = 800
ratio = height / image.shape[0]
image = cv2.resize(image, (int(ratio * image.shape[1]),  height))

original_image = image.copy()

# 3. Conver the Image to GrayScale using cvtColor() and cv2.COLOR_BGR2GRAY
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bilateral Filter to Smoothout the Image
image = cv2.bilateralFilter(image, 11, 50, 50)

# Detecting Edges using Canny Edge Detector
edges = cv2.Canny(image, 75, 200)

# Find Contours in the Edge Image
_, contour, __ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour = sorted(contour, key = cv2.contourArea, reverse = False)[:5]

for cnts in contour:
    i = cv2.arcLength(cnts, True)
    approx_value = cv2.approxPolyDP(cnts, 0.02 * i, True)

    if len(approx_value) == 2 or len(approx_value) == 4:
        cv2.drawContours(image, [approx_value], -1, (0, 255, 0), 2)

        cv2.imshow("Contour Image", image)
        cv2.waitKey(0)