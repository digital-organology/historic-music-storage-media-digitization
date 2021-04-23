import cv2
import numpy as np
import sys

def change_contrast_brightness(picture, erosion_kernel, noise_kernel, contrast_factor=1, brightness_val=0):
    # convert to gray
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    # threshold at high intensity
    cv2.imwrite("grayish.jpg", gray)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)#cramped up for metallplatten
    cv2.imwrite("blackwhite_240.jpg", thresh)
    #kernel = np.ones((3, 3), np.uint8)
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    #bigger_kernel = np.ones((5, 5), np.uint8)
    bigger_kernel = np.ones((3, 3), np.uint8)
    noise_removed = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, bigger_kernel)

    brighter_picture = np.where((255 - noise_removed) < brightness_val, 255, noise_removed + brightness_val)
    contrast_picture = brighter_picture * contrast_factor
    cv2.imwrite("contrast73_4_2.jpg", contrast_picture)

    return contrast_picture.astype(np.uint8)


def picture_to_blackwhite(picture):
    grayScale = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    (_, bwImage) = cv2.threshold(grayScale, 127, 255, cv2.THRESH_BINARY)
    return bwImage


def show_difference(picture_file1, picture_file2, outf, color_dif=True):
    picture1 = cv2.imread(picture_file1)
    picture2 = cv2.imread(picture_file2)
    if not color_dif:#disregard different coloring
        picture1 = picture_to_blackwhite(picture1)
        picture2 = picture_to_blackwhite(picture2)
    mask_equality = picture1 == picture2
    picture1[mask_equality] = 0
    cv2.imwrite(outf, picture1)
    #assert (picture1 == picture2).all()