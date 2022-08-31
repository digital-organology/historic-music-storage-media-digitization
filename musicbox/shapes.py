from scipy.spatial import distance
from itertools import compress
import numpy as np
import cv2
import musicbox.helpers
import timeit
import os

def find_disc_edges(proc):
    # We need to know what area the label is. The config gives us
    # the relative distance between label and center in relation to the
    # distance between center and disc edge.
    # So we need to know where the edge (of the disc, not the U2 member) is first.

    indices, counts = np.unique(proc.labels, return_counts = True)
    indices = indices[1:]
    counts = counts[1:]
    max_count = np.argmax(counts)
    edge_id = indices[max_count]

    proc.shapes.pop(edge_id, None)

    # We transform it into a list of coordinates (what we call shape around here)

    proc.disc_edge = np.argwhere(proc.labels == edge_id).astype(np.int32)

    # Now we can calculate where the label ends

    # As this is the first time we use anything scipy.spatial related (at least while I'm doing this rewrite)
    # let me address this here: The entire confusion around the ordering of coordinates in our code up until now
    # stems from one problem: scipy.spatial is intended for spatial data which uses lat, long coordinates.
    # This effectively means for our purposes that what scipy expects is y,x coordinate pairs...
    # Through some coincidence, this is also the way numpy handles image data, e.g. the shape
    # of one of our images is (4000, 6000). Since our images are wider than they are tall, this
    # means the first axis is actually the y axis...again. And since we use argwhere to generate
    # coordinates they are also in the format y,x. So this all works, but we have to keep it in mind

    edge_radius_calc = distance.cdist([(proc.center_y, proc.center_x)], proc.disc_edge).min()
    label_radius_calc = edge_radius_calc / proc.parameters["outer_radius"] * proc.parameters["inner_radius"]

    # Create a circle, this could surely be done more efficient

    bg = np.zeros_like(proc.labels).astype(np.uint8)
    circle = cv2.circle(bg, (proc.center_x, proc.center_y), round(label_radius_calc), 1)
    
    proc.label_edge = np.argwhere(circle == 1).astype(np.int32)

    return True

def filter_inner(proc):
    # This filters everything that is inside the inner label area.
    
    # We can calculate where the label ends

    edge_radius_calc = distance.cdist([(proc.center_y, proc.center_x)], proc.disc_edge).min()
    label_radius_calc = edge_radius_calc / proc.parameters["outer_radius"] * proc.parameters["inner_radius"]

    keep = []

    # Finally we can filter the shapes based on their distance to the center
    for shape in proc.shapes.values():
        # Is min the right thing to do here? I don't know.
        dist = distance.cdist(shape, [(proc.center_y, proc.center_x)]).min()
        if dist < label_radius_calc:
            keep.append(False)
        else:
            keep.append(True)

    keys = list(proc.shapes.keys())
    keys_to_keep = list(compress(keys, keep))

    filtered_shapes = { keep_shape: proc.shapes[keep_shape] for keep_shape in keys_to_keep }

    if "debug_dir" in proc.parameters.keys():
        start_time = timeit.default_timer()
        
        # Generate image prior to filtering
        shapes_before = musicbox.helpers.make_image_from_shapes(proc.current_image, proc.shapes)
        shapes_before = musicbox.helpers.make_color_image(shapes_before)
        shapes_before = cv2.circle(shapes_before, (proc.center_x, proc.center_y), 3, (255, 0, 0), 3)

        cv2.imwrite(os.path.join(proc.parameters["debug_dir"], "filter_label_before.tiff"), shapes_before)

        shapes_after = musicbox.helpers.make_image_from_shapes(proc.current_image, filtered_shapes)
        shapes_after = musicbox.helpers.make_color_image(shapes_after)
        cv2.imwrite(os.path.join(proc.parameters["debug_dir"], "filter_label_after.tiff"), shapes_after)
        
        print("INFO: Creating debug information added an overhead of " + ("%.5f" % (timeit.default_timer() - start_time)) + " seconds")

    proc.shapes = filtered_shapes

    return True