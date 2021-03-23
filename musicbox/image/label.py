import cv2
import numpy as np
from skimage import measure
from musicbox.helpers import gen_lut

def _find_connected_shapes(img, y, x, distance, y_limit, x_limit):
    # Define search area

    y_from = max(0, y - distance)
    y_to = min(y_limit - 1, y + distance) + 1
    
    # We do not need to look to the right (meaning anything with higher x than what we're looking at)
    x_from = max(0, x - distance)
    x_to = min(x_limit - 1, x) + 1

    # Select subarray that is our search radius
    search_space = img[y_from:y_to, x_from:x_to]
    
    # Get ids of surrounding shapes
    ids = np.unique(search_space)
    ids = ids[ids != -1]
    ids = ids[ids != 0]

    return ids

def _process_pixel(img, y, x, distance, y_limit, x_limit, current_shape):
    pixel_content = img[y,x]

    if pixel_content == 0:
        # Background pixel, no need to do anything
        return current_shape

    # Unprocessed pixel, will need to process

    # Get all surrounding shapes
    shape_ids = _find_connected_shapes(img, y, x, distance, y_limit, x_limit)

    if shape_ids.size == 0:
        # No close shape, start a new one
        current_shape = current_shape + 1
        img[y,x] = current_shape
        return current_shape

    if shape_ids.size == 1:
        # Only one shape is close, assign this pixel to that shape
        img[y,x] = shape_ids[0]
        return current_shape

    # Multiple shapes are close. We assign this pixel to the oldest
    # meaning the one with the smallest id, and also assign all other close shapes
    # to that shape

    oldest_shape = shape_ids.min()
    img[y,x] = oldest_shape

    for shape in shape_ids[shape_ids != oldest_shape]:
        img[img == shape] = oldest_shape

    return current_shape

def label_image(img, distance):
    """
    Will label all connected components of the provided image.
    There may be gaps in label ids due to specifics of the implementation.

    :Returns:
        labels : A 2d numpy array of the connected components 
        colored_image : A rgb colored representation of the same
    """        

    # Make sure every pixel either is marked as background or foreground
    img = img > 0

    img = img.astype(int)

    if distance == 1:
        img = measure.label(img, background = 0)
    else:
        # Multiply everything by -1 so that unprocessed pixels will be -1 while background is still 0
        img = img * -1

        y_limit = img.shape[0]
        x_limit = img.shape[1]

        # Iterate over every pixel once (does this make us yolo?)

        current_shape = 1

        for x_next in range(0, x_limit - 1):
            # print("Processing x:", x_next)
            for y_next in range(0, y_limit - 1):
                # print("Processing y:", y_next)
                current_shape = _process_pixel(img, y_next, x_next, distance, y_limit, x_limit, current_shape)


    # Get color table
    lut = gen_lut()

    # Make sure there are at max 256 labels
    labels = img.copy().astype(np.uint8)
    labels = cv2.LUT(cv2.merge((labels, labels, labels)), lut)
    return img, labels