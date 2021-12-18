import cv2
import numpy as np

def binarization(proc):
    _, img_threshold = cv2.threshold(proc.current_image, 60, 255, cv2.THRESH_BINARY)
    img_out = (img_threshold > 0).astype(np.uint8)
    proc.current_image = img_out
    return True

def edge_in(proc):
    image = (proc.current_image > 0).astype(np.uint8)

    kernel = np.ones((3,3), np.uint8)

    image_eroded = cv2.erode(image, kernel)

    edges = image - image_eroded

    proc.current_image = edges