from musicbox.helpers import gen_lut
from musicbox.helpers import calculate_angles
import numpy as np
import os
import sys
from scipy.spatial import distance
from sklearn.cluster import MeanShift
from itertools import compress
import cv2
import json
import alphashape
from scipy.spatial import distance
import cv2
import numpy as np
from musicbox.helpers import calculate_angles

from numpy import genfromtxt

def convert_notes(shapes_dict, center_x, center_y):
    shapes = shapes_dict.values()
    shape_ids = list(shapes_dict.keys())
    shape_min = []
    shape_max = []
    for shape in shapes:
        mini, maxi = calculate_angles(shape, center_x, center_y)
        shape_min.append(mini)
        shape_max.append(maxi)
    
    # import pdb; pdb.set_trace()
    arr = np.column_stack((list(shape_ids), shape_min, shape_max))
    diff = arr[:,2] - arr[:,1]
    # diff[diff > 200] = 360 - diff[diff > 200]
    arr = np.c_[arr, diff]
    return arr


def _fix_empty_tracks(data_array, first_track, track_width):
    mean_dists = []
    for i in np.unique(data_array[:,1]):
        vals = data_array[np.where(data_array[:,1] == i)]
        mean_dists.append([i, np.mean(vals[:,2])])

    # Convert to numpy array and insert dummy first row for the closest track to the center
    mean_dists = np.array(mean_dists)
    mean_dists = np.insert(mean_dists, 0, np.array((1, first_track)), 0)

    # Iterate over the array and do the following for each track:
    # Calculate the distance to the previous track and divide this by the average track width
    # Round this. Assign the track number of the previous track plus the result of the rounding.

    for i in range(1, mean_dists.shape[0]):
        dist = mean_dists[i, 1] - mean_dists[i - 1, 1]
        track_gap = round(dist / track_width)
        mean_dists[i, 0] = mean_dists[i - 1, 0] + track_gap

    # Now we replace the track values in the original array with the correct ones
    # As tracks start with 1 and we use these as index to our mappings we dont even need to remove the dummy first row :-)

    copy = data_array.copy()
    for track in np.unique(data_array[:,1]):
        copy[:,1][data_array[:,1] == track] = mean_dists[int(track), 0]

    copy = np.delete(copy, 2, 1)
    return copy



def extract_notes(img,
                    center_x,
                    center_y,
                    additional_arguments,
                    create_json = False):          
    # First of let us find the outer border
    # This should be the first found shape, meaning the one with the lowest index (apart from 0)

    # import pdb; pdb.set_trace()

    labels = img.copy()

    # indices = np.unique(labels)
    # outer_border_id = indices[indices != 0].min()
    indices, counts = np.unique(labels, return_counts = True)
    indices = indices[1:]
    counts = counts[1:]
    max_count = np.argmax(counts)
    outer_border_id = indices[max_count]
    
    # Transform it into a polygon

    outer_border = np.argwhere(labels == outer_border_id).astype(np.int32)

    # Switch x and y around
    outer_border[:,[0, 1]] = outer_border[:, [1, 0]]

    # if not use_punchhole:#for non-metal plates: updated center definition
    #     center_x, center_y = alternative_center(outer_border)

    # print("outer border center calc:", tuplex)
    # color_image = cv2.circle(img, (tuplex[0], tuplex[1]), 3, (255, 0, 0), 3)
    # cv2.imshow("ci", color_image)


    # To find the inner border we calculate the distance from the center
    # to the outer border. We can then calculate a approximate circular inner border.

    outer_radius_calc = distance.cdist([(center_x, center_y)], outer_border).min()
    inner_radius_calc = outer_radius_calc / additional_arguments["outer_radius"] * additional_arguments["inner_radius"]

    # Create a circle, this could surely be done more efficient

    bg = np.zeros_like(img).astype(np.uint8)
    circle = cv2.circle(bg, (center_x, center_y), round(inner_radius_calc), 1)
    
    inner_border = np.argwhere(circle == 1).astype(np.int32)
    inner_border[:,[0, 1]] = inner_border[:, [1, 0]]

    # Great, now we convert all found shapes to polygons
    # To do this, we filter them by size first

    unique, counts = np.unique(img, return_counts = True)
    
    median = np.median(counts)
    lower_bound = median / 3
    upper_bound = 8 * median

    shape_candidates = np.where(np.logical_and(counts >= lower_bound, counts <= upper_bound))
    shape_ids = unique[shape_candidates]

    # We convert them to polygons next and while we're at it find the center point for each one

    shapes = []
    centers = []
    for shape_id in shape_ids:
        contour = np.argwhere(img == shape_id).astype(np.uint32)
        contour[:, [0, 1]] = contour[:, [1, 0]]
        if additional_arguments["use_punchholes"]:
            mini, maxi = calculate_angles(contour, center_x, center_y, return_points = True)
            if additional_arguments["punchhole_side"] == "left":
                #mini = np.array([mini[1], mini[0]])
                centers.append(mini.astype(np.uint32))
            elif additional_arguments["punchhole_side"] == "right":
                #maxi = np.array([maxi[1], maxi[0]])
                centers.append(maxi.astype(np.uint32))
            else:
                raise Exception("You donut that is not a side")
        else:
            point = np.mean(contour, 0).astype(np.uint32)
            centers.append(point)
        shapes.append(contour)

    # We can use the center points to determine if the shape are inside the inner
    # border or outside
    # We calculate the distance to the center, if it is smaller than the inner radius we drop the point
    # otherwise we keep it

    keep = []
    for point in centers:
        center_dist = distance.cdist([point], [(center_x, center_y)])[0][0]
        if center_dist < inner_radius_calc:
            keep.append(False)
        else:
            keep.append(True)
    
    shape_ids = shape_ids[keep]
    shapes = list(compress(shapes, keep))
    centers = list(compress(centers, keep))

    # We can now go ahead and calculate the distances between each point and the outer/inner border
    # As we might have some shapes that are incomplete (but are notes we want to match correctly)
    # we also employ a mechanism to calculate their position that does not rely on correctly identifies center points

    outer_distances = [] # We do not actually use this right now
    inner_distances = []

    for point in centers:
        pnt = [(point[0], point[1])]
        #pnt = [(point[1], point[0])]
        # inner_distance = distance.cdist(pnt, inner_border).min()
        inner_distance = distance.cdist(pnt, [(center_x, center_y)]).min()
        outer_distance = distance.cdist(pnt, outer_border).min()
        sum_distance = outer_distance + inner_distance
        outer_distances.append(outer_distance / sum_distance)
        inner_distances.append(inner_distance / sum_distance)

    # Now we may start clustering these points

    data = np.array(inner_distances)
    data = data * 1000
    data = np.column_stack((data, np.zeros(len(data))))
    data = data.astype(int)
    ms = MeanShift(bandwidth = additional_arguments["bandwidth"], bin_seeding = True)
    ms.fit(data)
    classes = ms.labels_
    
    # We now go ahead and sort the clusters in ascending order beginning on the inside

    inner_assignments = np.column_stack((classes, np.array(inner_distances)))
    cluster_ids = np.unique(inner_assignments[:,0])
    means = []
    for cluster in cluster_ids:
        tmp = inner_assignments[np.where(inner_assignments[:,0] == cluster)]
        means.append(np.mean(tmp[:,1]))

    cluster_means = np.column_stack((cluster_ids, means))
    cluster_means = cluster_means[cluster_means[:,1].argsort()]

    # Magic. Don't touch.
    # Change old cluster assignments to the ones
    # sorted by distance from the inner circle
    # thus creating a hierarchically sorted order 
    copy = np.zeros_like(classes)
    for cluster in np.unique(classes):
        copy[classes == cluster] = np.argwhere(cluster_means[:,0] == cluster)[0][0]

    classes = copy + 1

    # Fix empty tracks

    data_array = np.column_stack((shape_ids, classes, inner_distances))

    assignments = _fix_empty_tracks(data_array, additional_arguments["first_track"], additional_arguments["track_width"])

    # Create colored output

    image = img.copy()
    mask = np.isin(image, shape_ids)
    mask = np.invert(mask)
    np.putmask(image, mask, 0)

    inner_border_label = max(classes) + 1
    outer_border_label = inner_border_label + 1

    image[inner_border[:,1], inner_border[:,0]] = inner_border_label

    image[outer_border[:,1], outer_border[:,0]] = outer_border_label

    # Does not work but would probably be much faster :(
    # assignment_dict = zip(self._shape_ids, assignments)
    # image = [assignment_dict[i] for i in image]

    assignment_dict = dict(zip(assignments[:,0], assignments[:,1]))
    image = np.vectorize(assignment_dict.get)(image)
    image[image == None] = 0

    lut = gen_lut()

    color_image = image.astype(np.uint8)
    color_image = cv2.LUT(cv2.merge((color_image, color_image, color_image)), lut)
    color_image = cv2.circle(color_image, (center_x, center_y), 3, (255, 0, 0), 3)

    for center in centers:
        color_image[center[1], center[0]] = (255, 255, 255)
    
    shapes_dict = dict(zip(shape_ids, shapes))

    annotated_image = color_image.copy()

    if "debug_dir" in additional_arguments.keys():
        # If you want to look at clustering results:
        # cluster_data = genfromtxt("dataaar.csv", delimiter = ",")
        for i in range(len(centers)):
            assigned_id = assignments[i, 1]#0 -> shape_id, 1 -> track_id
            # assigned_id = cluster_data[i, 1]#0 -> shape_id, 1 -> track_id
            point = centers[i]
            point = (point[0], point[1])
            cv2.putText(annotated_image,
                        str(assigned_id),
                        point,
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)

        cv2.imwrite(os.path.join(additional_arguments["debug_dir"], "numbers_debug.tiff"), annotated_image)


        # import pdb; pdb.set_trace()

        debug_array = np.column_stack((shape_ids, classes, assignments[:,1], inner_distances))
        np.savetxt(os.path.join(additional_arguments["debug_dir"], "debug.txt"), debug_array, delimiter = ",", fmt= "%1.5f")

        cv2.imwrite(os.path.join(additional_arguments["debug_dir"], "tracks.tiff"), color_image)

    if create_json:
        print(sys.path[0])
        output = {}
        json_data = []
        for i in range(len(shape_ids)):
            shape_data = {}
            shape_data["id"] = int(shape_ids[i])
            shape_data["track"] = int(assignments[i, 1])
            # import pdb; pdb.set_trace()
            ashape = alphashape.alphashape(shapes[i].tolist(), 0.)
            shape_data["points"] = "M" + "L".join([str(x) + "," + str(y) for (x, y) in np.asarray(ashape.exterior.coords).tolist()]) + "Z"
            # import pdb; pdb.set_trace()
            json_data.append(shape_data)
        output["data"] = json_data
        output["center"] = [center_x, center_y, inner_radius_calc]
        with open(os.path.join(sys.path[0], "client", "data.json"), "w", encoding="utf8") as f:
            json.dump(output, f, indent = 4)


    return shapes_dict, assignments, color_image
