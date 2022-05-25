import os
import cv2 as cv

image_dir = r'D:\TextOCR\splited\train\image'
for item in os.listdir(image_dir):
    image = cv.imread(os.path.join(image_dir, item))
    assert max(image.shape) <= 1024, image.shape
    print(image.shape)