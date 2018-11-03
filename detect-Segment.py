# Libraries
import cv2
import imutils
import numpy as np

# Path to the Image
image_path = "../DataSet/DIQ_Part1/set12/2012-04-19_18-27-07_716.jpg"

# Steps to Detect Document in the Image
#    1. Load the Image and resize it.
#    2. Convert the Image to GrayScale.
#    3. Image Process
#       3.1. Bilaternal Filter to reduce Noise and smooth out image.
#       3.2. Adaptive Threshold to Convert the Image to an Binary Image
#       3.3. Median Blur to filter out small pixel details
#       3.4. Black Border around Image to handle Documents that maybe touching the Border of Image
#    4. Detect Edges using Canny Edge Detector.
#    5. Find Contours in the Edge Detected Image.
#    6. Pick Points which Cover the Document.
#    7. Transform the Image

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

# Adaptive Threshold to remove noise from the image 
# It would tranform the image into an binary image
# Documentation at
#   https://docs.opencv.org/3.4.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

# Median filter would help in filtering or cleaning small details from the image
image = cv2.medianBlur(image, 11)

# There are some images where the document is touching the image border
# To handle such images
# Add a black border around the image
image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value = [0, 0, 0])
# Adds a Black Border of 4 pixels in all direction

# Detecting Edges using Canny Edge Detector
edges = cv2.Canny(image, 200, 250)

# Find Contours in the Edge Image
_, contour, __ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour = sorted(contour, key = cv2.contourArea, reverse = True)[:10]

# for each contour
for cnts in contour:

    # Find the Perimeter
    i = cv2.arcLength(cnts, True)
    # Approximate the Curve
    approx_value = cv2.approxPolyDP(cnts, 0.03 * i, True)

    # Assuming Document has 4 Corners
    if len(approx_value) == 4:
        points = approx_value
        break;

# Find Corner Points i.e Top-left, Top-right, Bottom-left, Bottom-right
pts = points
points = points.reshape(4, 2)

diff = np.diff(points, axis=1)
summ = points.sum(axis=1)

# Top
# Top-left have the Smalles Sum and Bottom-right have the maximum summ

# Bottom
# Top-right have the minimum difference and Bottom-left have the maximum difference.
rectange = np.array([points[np.argmin(summ)], points[np.argmax(diff)], points[np.argmax(summ)], points[np.argmin(diff)]])

# Offset Contour, by 4px border
offset = (-4, -4)       
rectange += offset	    	       
rectange[rectange < 0] = 0

# Rescale the Image to Original
rescaled = pts.dot(original_image.shape[0] / height)

# Calculate the Height and Width of New Image
height = max(np.linalg.norm(rescaled[0] - rescaled[1]), np.linalg.norm(rescaled[2] - rescaled[3]))
width = max(np.linalg.norm(rescaled[1] - rescaled[2]), np.linalg.norm(rescaled[3] - rescaled[0]))

# Target Points to View the New Image from correct angel.
target = np.array([[0, 0],[0, height],[width, height],[width, 0]], np.float32)

# Convert rescaled image to float32
rescaled = rescaled.astype(np.float32)

# Get the Perspective
M = cv2.getPerspectiveTransform(rescaled, target)

# Wrap the Image to get a new Image
output = cv2.warpPerspective(original_image, M, (int(width), int(height)))

# Display the Output Image
cv2.imshow("Original Image", original_image)
cv2.imshow("Output", output)
cv2.waitKey(0)