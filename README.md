# Inkredo-Programming-Challenge

## Task

  Efficiently detect and segment document regions.
  
  For this challenge,<br>
    **_Input_** consists in a set of files containing a document from a predefined set <br>
    **_Output_** should be cropped images in which one can find the document without any other noise for extracting data. 

---

## Detection and Segmentataion of Documents in Images.

### Steps to Detect Document in the Image
    1. Load the Image and resize it.
    2. Convert the Image to GrayScale.
    3. Image Process
        3.1. Bilaternal Filter to reduce Noise and smooth out image.
        3.2. Adaptive Threshold to Convert the Image to an Binary Image
        3.3. Median Blur to filter out small pixel details
        3.4. Black Border around Image to handle Documents that maybe touching the Border of Image
    4. Detect Edges using Canny Edge Detector.
    5. Find Contours in the Edge Detected Image.
    6. Pick Points which Cover the Document.
    7. Transform the Image

---
## Installation
```sh
  git clone https://github.com/Yashs744/Inkredo-Programming-Challenge.git
  cd Inkredo-Programming-Challenge/
```

### Requirements
```sh
  pip3 install -r requirements.txt
```

### Execution
```sh
  python3 detect-Segment.py --image path/to/the/image.jpg
```
---

## Example
![result-1](https://github.com/Yashs744/Inkredo-Programming-Challenge/blob/master/result/result-1.png)

### Input
![result-2(i)](https://github.com/Yashs744/Inkredo-Programming-Challenge/blob/master/result/result-2(i).png)
### Output
![result-2(ii)](https://github.com/Yashs744/Inkredo-Programming-Challenge/blob/master/result/result-2(Ii).png)

---

## What's Next ?
To get results close to human level cropping and segmentation with minimum error rate we can use _Auto-Encoder Network_.<br>
We can create a dataset of images by cropping the document manually then the network would work as follow:

  - Pass the Image to Encoder part of the Network.
  - Expect a Cropped Document from the Network as an Output.
  - Minimize the Loss between the Document produced by the Network and manually Segmented Document.

---

## References
  - [PyImageSearch](https://www.pyimagesearch.com)
  - [OpenCV Documentation](https://docs.opencv.org/3.4.3/)
