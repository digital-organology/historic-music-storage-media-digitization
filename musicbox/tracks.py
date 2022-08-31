from scipy.spatial import distance
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import numpy as np
import timeit
import cv2
import os
import musicbox.helpers

def mean_shift(proc):
    # TODO: We could possibly work with the absolute distance to the outer border which would be affected by warping but not by a wrong center
    inner_distances = []

    for id, shape in proc.shapes.items():
        shape_center = np.mean(shape, 0).astype(np.uint16)
        shape_center = [(shape_center[0], shape_center[1])]
        inner_distance = distance.cdist(shape_center, [(proc.center_y, proc.center_x)]).min()
        outer_distance = distance.cdist(shape_center, proc.disc_edge).min()
        inner_distances.append(inner_distance / (outer_distance + inner_distance))


    if "debug_dir" in proc.parameters:
        start_time = timeit.default_timer()

        plt.scatter(inner_distances, np.zeros_like(inner_distances))
        plt.savefig(os.path.join(proc.parameters["debug_dir"], "note_distances.tiff"))

        print("INFO: Creating debug information added an overhead of " + ("%.5f" % (timeit.default_timer() - start_time)) + " seconds")

    # Now we may start clustering these points

    data = np.array(inner_distances)
    data = data * 1000
    data = np.column_stack((data, np.zeros(len(data))))
    data = data.astype(int)
    ms = MeanShift(bandwidth = proc.parameters["bandwidth"], bin_seeding = True)
    ms.fit(data)
    classes = ms.labels_
    

    # We now go ahead and sort the clusters in ascending order beginning on the inside
    # We first calculate the mean distance of each cluster
    # This is fairly similar (to at least 3 decimal places) to ms.cluster_centers_[:,0] / 1000
    # so we could possibly just use that

    assignments = np.column_stack((classes, np.array(inner_distances)))

    ids = np.unique(assignments[:,0])
    cluster_means = [np.mean(assignments[assignments[:,0] == i, 1]) for i in ids]

    cluster_means = np.column_stack((ids, cluster_means))
    cluster_means = cluster_means[cluster_means[:,1].argsort()]

    # Magic. Don't touch.
    # Change old cluster assignments to the ones
    # sorted by distance from the inner circle
    # thus creating a hierarchically sorted order
    # We could do this in a more efficient way possible (using a dictionary)
    # but as the entire thing takes about .3 milliseconds it's probably not worth the trouble
    copy = np.zeros_like(classes)
    for cluster in np.unique(classes):
        copy[classes == cluster] = np.argwhere(cluster_means[:,0] == cluster)[0][0]

    classes = copy + 1

    proc.assignments = dict(zip(proc.shapes.keys(), classes))

    proc.track_distances = np.column_stack((np.arange(1, cluster_means.shape[0] + 1), cluster_means[:,1]))

    # import pdb; pdb.set_trace()

    if "debug_dir" in proc.parameters.keys():
        start_time = timeit.default_timer()

        # This could be done faster
        plot_data = zip(classes, proc.shapes.values())
        tracks_image = musicbox.helpers.make_image_from_shapes(proc.current_image, plot_data)
        tracks_image = musicbox.helpers.make_color_image(tracks_image)
        tracks_image = cv2.circle(tracks_image, (proc.center_x, proc.center_y), 3, (255, 0, 0), 3)

        cv2.imwrite(os.path.join(proc.parameters["debug_dir"], "tracks.tiff"), tracks_image)

        print("INFO: Creating debug information added an overhead of " + ("%.5f" % (timeit.default_timer() - start_time)) + " seconds")

    return True

def correct_empty(proc):
    mean_dists = proc.track_distances.copy()
    mean_dists = np.insert(mean_dists, 0, np.array((0, proc.parameters["first_track"])), 0)

    track_dists = np.diff(mean_dists, axis=0)
    track_dists = np.column_stack((mean_dists[:,0], np.append(0, track_dists[:,1])))
    m_track_width = np.median(track_dists, axis = 0)[1]

    while np.max(mean_dists, axis = 0)[0] < proc.parameters["n_tracks"]:
        max_row_idx = np.argmax(track_dists, axis = 0)[1]
        max_row = track_dists[max_row_idx]

        if round((max_row[1] - m_track_width) / m_track_width) < 1:
            break
        
        track_dists[max_row_idx][1] = max_row[1] - m_track_width

        # Increment all rows after this one up

        for i in range(max_row_idx, len(track_dists)):
            track_dists[i][0] = track_dists[i][0] + 1

    assignments = np.array(list(proc.assignments.items()))
    copy = assignments.copy()

    for track in np.unique(assignments[:,1]):
        copy[:,1][assignments[:,1] == track] = track_dists[int(track), 0]
    return True

