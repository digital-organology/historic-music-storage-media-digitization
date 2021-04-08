from __future__ import print_function
from __future__ import division
import cv2
import argparse
import numpy as np

slider_bright_name = 'Brightness Scale'
slider_contrast_name = 'Contrast Factor/100'
identifier = 'Parameter Search'

def change_contrast_brightness(picture, contrast_factor=1.5, brightness_val=25):
    brighter_picture = np.where((255 - picture) < brightness_val, 255, picture + brightness_val)
    contrast_picture = brighter_picture * contrast_factor
    return contrast_picture.astype(np.uint8)

def on_trackbar(val):
    brightness = cv2.getTrackbarPos(slider_bright_name, identifier)
    contrast = cv2.getTrackbarPos(slider_contrast_name, identifier)
    img = change_contrast_brightness(src, contrast / 100.0, brightness)
    print(brightness, contrast / 100.0)# it's a mac os thing...
    cv2.imshow('Processed Image', img)

parser = argparse.ArgumentParser(description='nope')
parser.add_argument('input', help='Path to input image', default='LinuxLogo.jpg')
args = parser.parse_args()

src = cv2.imread(cv2.samples.findFile(args.input))


cv2.namedWindow(identifier)
cv2.resizeWindow(identifier, 800, 600)

cv2.createTrackbar(slider_bright_name, identifier, 0, 100, on_trackbar)
cv2.createTrackbar(slider_contrast_name, identifier, -500, 500, on_trackbar)
# init
on_trackbar(1)
cv2.waitKey(0)

