import numpy as np
import matplotlib.pyplot as plt 
import cv2

def gen_lut():
        """
        Generate a label colormap compatible with opencv lookup table, based on
        Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
        appendix C2 `Pseudocolor Generation`.
        :Returns:
            color_lut : opencv compatible color lookup table
        """

        # Blatantly stolen from here: https://stackoverflow.com/a/57080906/3176892

        tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
        arr = np.arange(256)
        r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
        g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
        b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
        return np.concatenate([[[b]], [[g]], [[r]]]).T

def plot_polygon(pic, polygon):
    bg_image = np.zeros_like(pic).astype(np.uint8)
    bg_image[polygon[:,1], polygon[:,0]] = 1
    plt.imshow(bg_image)
    plt.show()


def change_contrast_brightness(picture, contrast_factor=1, brightness_val=0):
    # convert to gray
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    # threshold at high intensity
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    bigger_kernel = np.ones((5, 5), np.uint8)
    noise_removed = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, bigger_kernel)

    brighter_picture = np.where((255 - noise_removed) < brightness_val, 255, noise_removed + brightness_val)
    contrast_picture = brighter_picture * contrast_factor
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