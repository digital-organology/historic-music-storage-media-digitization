import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import pkgutil
import csv
import musicbox.data
from importlib_resources import files, as_file
from scipy.spatial import distance

def get_lut():
    """Loads a lut included in the package that can be used to generate a color image out of an 8-Bit image

    Returns:
        _type_: _description_
    """

    with files(musicbox.data).joinpath("lut.npy").open("rb") as f:
        lut = np.load(f)

    return lut

def make_image_from_shapes(canvas, shapes):
    if isinstance(shapes, dict):
        shapes = zip(shapes.keys(), shapes.values())

    bg_image = np.zeros_like(canvas).astype(np.uint16)

    for id, shape in shapes:
        bg_image[shape[:,0], shape[:,1]] = id
    return bg_image

def make_color_image(img):
    """Creates a colored image from an 8-Bit grayscale image

    Args:
        img (numpy.ndarray): Array of the grayscale image to color

    Returns:
        numpy.ndarray: Color representation of the provided grayscale image
    """

    lut = get_lut()

    color_image = img.astype(np.uint8)
    color_image = cv2.LUT(cv2.merge((color_image, color_image, color_image)), lut)

    return color_image

def plot_polygon(pic, polygon):
    bg_image = np.zeros_like(pic).astype(np.uint8)
    bg_image[polygon[:,0], polygon[:,1]] = 1
    plt.imshow(bg_image)
    plt.show()

def midi_to_notes():
    stream = pkgutil.get_data(__name__, "data/midi_notes.csv")
    reader = csv.reader(stream.decode("utf-8").splitlines(), delimiter = ",")
    return {int(row[0]):row[1] for row in reader}

def enter_debug(proc):
    # Here as an debug entry point
    pass