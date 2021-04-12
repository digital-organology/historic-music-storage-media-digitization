from musicbox.helpers import gen_lut
import musicbox.image.label
import numpy as np
import math
from scipy.spatial import distance
from sklearn.cluster import MeanShift
from itertools import compress
import cv2

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



def extract_notes(img, outer_radius, inner_radius, center_x, center_y, bwidth, first_track, track_width, img_grayscale, compat_mode = False, exact_mode = False):
        # img_grayscale is only needed when compat_mode is set to True.
        # We will use it to run another pass detecting shapes with a high search area to find the outermost border

    # We can now go ahead and calculate the distances between each point and the outer/inner border

        if compat_mode:
            labels, labels_color = musicbox.image.label.label_image(img_grayscale, 5)
        else:
            labels = img.copy()

        indices = np.unique(labels)
        outer_border_id = indices[indices != 0].min()
        
        # Transform it into a polygon

        outer_border = np.argwhere(labels == outer_border_id).astype(np.int32)

        # Switch x and y around
        outer_border[:,[0, 1]] = outer_border[:, [1, 0]]

        # To find the inner border we calculate the distance from the center
        # to the outer border. We can then calculate a approximate circular inner border.

        outer_radius_calc = distance.cdist([(center_x, center_y)], outer_border).min()
        inner_radius_calc = outer_radius_calc / outer_radius * inner_radius

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
        upper_bound = 3 * median

        shape_candidates = np.where(np.logical_and(counts >= lower_bound, counts <= upper_bound))
        shape_ids = unique[shape_candidates]

        # We convert them to polygons next and while we're at it find the center point for each one

        shapes = []
        centers = []
        for shape_id in shape_ids:
            contour = np.argwhere(img == shape_id).astype(np.uint32)
            point = np.mean(contour, 0).astype(np.uint32)
            point = np.array([point[1], point[0]])
            contour[:, [0, 1]] = contour[:, [1, 0]]
            shapes.append(contour)
            centers.append(point)

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

        if exact_mode:
            for shape in shapes:
                inner_distance = distance.cdist(shape, [(center_x, center_y)]).min()
                outer_distance = distance.cdist(shape, outer_border).min()
                sum_distance = outer_distance + inner_distance
                outer_distances.append(outer_distance / sum_distance)
                inner_distances.append(inner_distance / sum_distance)
        else:
            for point in centers:
                pnt = [(point[0], point[1])]
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
        ms = MeanShift(bandwidth = bwidth, bin_seeding = True)
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

    return shapes_dict, assignments, color_image
