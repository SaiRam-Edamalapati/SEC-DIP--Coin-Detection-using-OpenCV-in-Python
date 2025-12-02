## WORKSHOP -4: Coin Detection using OpenCV in Python
## Name: Sai Ram E
## Register no: 212224240141
## AIM:
To perform Gray scale Morphology Real Time Bone Fracture Detection.

## PROGRAM:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
```
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred
```
```
def detect_fractures(preprocessed, original):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(preprocessed, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    edges = cv2.Canny(dilation, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = original.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result
```
```
def present_results(original_image, processed_image):
    # Convert from BGR (OpenCV) to RGB (Matplotlib)
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
 processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
```
```
 plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Fracture Detected Image")
    plt.imshow(processed_rgb)
    plt.axis('off')
    plt.show()
```
```
image_path = 'image.png'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the file path.")
else:
    preprocessed = preprocess_image(image)
    fracture_detected_image = detect_fractures(preprocessed, image)
    present_results(image, fracture_detected_image)
```

## OUTPUT:

<img width="1210" height="303" alt="image" src="https://github.com/user-attachments/assets/2aa99c84-9522-47bd-b35f-915f65ada44a" />
