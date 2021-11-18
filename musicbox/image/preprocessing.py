import cv2
import numpy as np
import sys

def preprocess(method: str, img: np.ndarray, additional_parameters = dict()):
    """Main method for image preprocessing  will dispatch the desired method

    Args:
        method (str): Method to use for labeling, currently basic (mostly binarization) and full (not implemented yet) are available
        img (numpy.ndarray): Image to label, needs to be binarized or at least have its background be 0
        additional_parameters (dict, optional): Dictionary of optional arguments. Defaults to dict().

    Raises:
        TypeError: Raised if invalid input data is provided
        ValueError: Raised if an invalid method for labeling is given

    Returns:
        numpy.ndarray: Image with labeled components
    """

    # Basic parameter checking
    if not isinstance(method, str):
        raise TypeError("method is not a string, aborting")
    
    # See if we got a valid parameter
    available_methods = ["basic", "full"]
    if not any(method in s for s in available_methods):
        raise ValueError("Method " + method + " is not implemented")

    # Call apropiate method
    if method == "basic":
        return preprocess_basic(img, additional_parameters)
    if method == "full":
        return preprocess_full(img, additional_parameters)

    return None

def preprocess_basic(img, additional_parameters):
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_threshold = cv2.threshold(img_grayscale, 60, 255, cv2.THRESH_BINARY)
    img_out = (img_threshold > 0).astype(int)
    return img_out

def preprocess_full(img, additional_parameters):
    raise NotImplementedError

# Unused code written by some person no longer working here...

# def change_contrast_brightness(picture, erosion_kernel, noise_kernel, thresh_binary, contrast_factor=1, brightness_val=0):
#     # convert to gray
#     gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
#     # threshold at high intensity
#     #cv2.imwrite("grayish.jpg", gray)
#     _, thresh = cv2.threshold(gray, thresh_binary, 255, cv2.THRESH_BINARY)#cramped up for metallplatten
#     #cv2.imwrite("blackwhite_240.jpg", thresh)
#     kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)
#     erosion = cv2.erode(thresh, kernel, iterations=1)
#     bigger_kernel = np.ones((noise_kernel, noise_kernel), np.uint8)
#     noise_removed = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, bigger_kernel)

#     brighter_picture = np.where((255 - noise_removed) < brightness_val, 255, noise_removed + brightness_val)
#     contrast_picture = brighter_picture * contrast_factor
#     #cv2.imwrite("contrast73_4_2.jpg", contrast_picture)

#     return contrast_picture.astype(np.uint8)


# def picture_to_blackwhite(picture):
#     grayScale = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
#     (_, bwImage) = cv2.threshold(grayScale, 127, 255, cv2.THRESH_BINARY)
#     return bwImage


# def show_difference(picture_file1, picture_file2, outf, color_dif=True):
#     picture1 = cv2.imread(picture_file1)
#     picture2 = cv2.imread(picture_file2)
#     if not color_dif:#disregard different coloring
#         picture1 = picture_to_blackwhite(picture1)
#         picture2 = picture_to_blackwhite(picture2)
#     mask_equality = picture1 == picture2
#     picture1[mask_equality] = 0
#     cv2.imwrite(outf, picture1)
#     #assert (picture1 == picture2).all()