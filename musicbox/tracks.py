from scipy.spatial import distance
from sklearn.cluster import MeanShift
import numpy as np

def mean_shift(proc):
    # TODO: We could possibly work with the absolute distance to the outer border which would be affected by warping but not by a wrong center
    inner_distances = []

    for id, shape in proc.shapes.items():
        shape_center = np.mean(shape, 0).astype(np.uint16)
        shape_center = [(shape_center[0], shape_center[1])]
        inner_distance = distance.cdist(shape_center, [(proc.center_y, proc.center_x)]).min()
        outer_distance = distance.cdist(shape_center, proc.disc_edge).min()
        inner_distances.append(inner_distance / (outer_distance + inner_distance))

    # Now we may start clustering these points

    data = np.array(inner_distances)
    data = data * 1000
    data = np.column_stack((data, np.zeros(len(data))))
    data = data.astype(int)
    ms = MeanShift(bandwidth = proc.parameters["bandwidth"], bin_seeding = True)
    ms.fit(data)
    classes = ms.labels_
    
    import pdb; pdb.set_trace()

    # This part could most likely be optimised

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