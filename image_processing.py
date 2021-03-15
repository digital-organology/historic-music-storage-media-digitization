#!/usr/bin/env python

import numpy as np
import cv2
import argparse
import math
import yaml
import sys
from midiutil.MidiFile import MIDIFile
from scipy.spatial import distance
from sklearn.cluster import MeanShift
from itertools import compress
# Only needed to use helper method plot_polygon:
import matplotlib.pyplot as plt 

class image_processor(object):
    _search_distance = 0
    _y_limit = 0
    _x_limit = 0
    _image = None
    _current_shape = 1
    _is_labeled = False

    def __init__(self, image, distance):
        self._image = image
        self._search_distance = distance

    def _find_connected_shapes(self, y, x):
        # Define search area

        y_from = max(0, y - self._search_distance)
        y_to = min(self._y_limit - 1, y + self._search_distance) + 1
        
        # We do not need to look to the right (meaning anything with higher x than what we're looking at)
        x_from = max(0, x - self._search_distance)
        x_to = min(self._x_limit - 1, x) + 1

        # Select subarray that is our search radius
        search_space = self._image[y_from:y_to, x_from:x_to]
        
        # Get ids of surrounding shapes
        ids = np.unique(search_space)
        ids = ids[ids != -1]
        ids = ids[ids != 0]

        return ids

    def _process_pixel(self, y, x):
        pixel_content = self._image[y,x]

        if pixel_content == 0:
            # Background pixel, no need to do anything
            return

        # Unprocessed pixel, will need to process

        # Get all surrounding shapes
        shape_ids = self._find_connected_shapes(y, x)

        if shape_ids.size == 0:
            # No close shape, start a new one
            self._current_shape = self._current_shape + 1
            self._image[y,x] = self._current_shape
            return

        if shape_ids.size == 1:
            # Only one shape is close, assign this pixel to that shape
            self._image[y,x] = shape_ids[0]
            return

        # Multiple shapes are close. We assign this pixel to the oldest
        # meaning the one with the smallest id, and also assign all other close shapes
        # to that shape

        oldest_shape = shape_ids.min()
        self._image[y,x] = oldest_shape

        for shape in shape_ids[shape_ids != oldest_shape]:
            self._image[self._image == shape] = oldest_shape

        return

    def _gen_lut(self):
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

    def label(self):
        """
        Will label all connected components of the provided image.
        There may be gaps in label ids due to specifics of the implementation.

        :Returns:
            labels : A 2d numpy array of the connected components 
            colored_image : A rgb colored representation of the same
        """        

        if self._is_labeled:
            # Get color table
            lut = self._gen_lut()

            # Make sure there are at max 256 labels
            labels = self._image.copy().astype(np.uint8)
            labels = cv2.LUT(cv2.merge((labels, labels, labels)), lut)
            return self._image, labels

        # Make sure every pixel either is marked as background or foreground
        self._image = self._image > 0

        self._image = self._image.astype(int)

        # Multiply everything by -1 so that unprocessed pixels will be -1 while background is still 0
        self._image = self._image * -1

        self._y_limit = self._image.shape[0]
        self._x_limit = self._image.shape[1]

        # Iterate over every pixel once (does this make us yolo?)

        for x_next in range(0, self._x_limit - 1):
            # print("Processing x:", x_next)
            for y_next in range(0, self._y_limit - 1):
                # print("Processing y:", y_next)
                self._process_pixel(y_next, x_next)

        self._is_labeled = True

        # Get color table
        lut = self._gen_lut()

        # Make sure there are at max 256 labels
        labels = self._image.copy().astype(np.uint8)
        labels = cv2.LUT(cv2.merge((labels, labels, labels)), lut)
        return self._image, labels

    def extract_shapes(self, outer_radius, inner_radius, center_x, center_y, bwidth):
        if not self._is_labeled:
            self.label()

        # First of let us find the outer border
        # This should be the first found shape, meaning the one with the lowest index (apart from 0)

        indices = np.unique(self._image)
        outer_border_id = indices[indices != 0].min()
        
        # Transform it into a polygon

        outer_border = np.argwhere(self._image == outer_border_id).astype(np.int32)

        # Switch x and y around
        outer_border[:,[0, 1]] = outer_border[:, [1, 0]]

        # To find the inner border we calculate the distance from the center
        # to the outer border. We can then calculate a approximate circular inner border.

        outer_radius_calc = distance.cdist([(center_x, center_y)], outer_border).min()
        inner_radius_calc = outer_radius_calc / outer_radius * inner_radius

        # Create a circle, this could surely be done more efficient

        bg = np.zeros_like(self._image).astype(np.uint8)
        circle = cv2.circle(bg, (center_x, center_y), round(inner_radius_calc), 1)
        
        inner_border = np.argwhere(circle == 1).astype(np.int32)
        inner_border[:,[0, 1]] = inner_border[:, [1, 0]]

        # Great, now we convert all found shapes to polygons
        # To do this, we filter them by size first

        unique, counts = np.unique(self._image, return_counts = True)
        
        median = np.median(counts)
        lower_bound = median / 3
        upper_bound = 3 * median

        shape_candidates = np.where(np.logical_and(counts >= lower_bound, counts <= upper_bound))
        shape_ids = unique[shape_candidates]

        # We convert them to polygons next and while we're at it find the center point for each one

        shapes = []
        centers = []
        for shape_id in shape_ids:
            contour = np.argwhere(self._image == shape_id).astype(np.uint32)
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

        outer_distances = [] # We do not actually use this right now
        inner_distances = []
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

        # Magic. Don't touch.
        # Change old cluster assignments to the ones
        # sorted by distance from the inner circle
        # thus creating a hierarchically sorted order 
        copy = np.zeros_like(classes)
        for cluster in np.unique(classes):
            copy[classes == cluster] = np.argwhere(cluster_means[:,0] == cluster)[0][0]

        classes = copy + 1

        assignments = np.column_stack((shape_ids, classes))

        # Create colored output

        image = self._image.copy()
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

        for shape in shape_ids:
            index = np.where(shape_ids == shape)
            image[image == shape] = classes[index]

        lut = self._gen_lut()

        color_image = image.astype(np.uint8)
        color_image = cv2.LUT(cv2.merge((color_image, color_image, color_image)), lut)

        shapes_dict = dict(zip(shape_ids, shapes))

        return shapes_dict, assignments, color_image

def plot_polygon(pic, polygon):
    bg_image = np.zeros_like(pic).astype(np.uint8)
    bg_image[polygon[:,1], polygon[:,0]] = 1
    plt.imshow(bg_image)
    plt.show()

class time_segmenter(object):
    _center_x = 2900
    _center_y = 2006
    _shapes = None
    _shape_ids = None

    def __init__(self, shapes, shape_ids, x, y):
        self._shape_ids = shape_ids
        self._center_x = x
        self._center_y = y
        self._shapes = shapes

    def _process_shape(self, shape):
        # shape[:,[0, 1]] = shape[:, [1, 0]]
        rectangle = cv2.minAreaRect(shape.astype(np.float32))
        box = cv2.boxPoints(rectangle)
        box = box.astype(np.uint32)

        if len(box) != 4:
            print(len(box))
            return (0, 0)

        dists = distance.cdist(box, [[self._center_x, self._center_y]])

        dists = dists.reshape(-1)
        idx = np.argpartition(dists, 2)

        closer_points = box[idx[:2]]
        other_points = box[idx[2:]]

        # print(closer_points)
        # print(other_points)

        dists_interpoint = distance.cdist(closer_points, other_points)

        # print(dists_interpoint)

        first_line = np.row_stack((closer_points[0], other_points[dists_interpoint[0].argmin()]))
        second_line = np.row_stack((closer_points[1], other_points[dists_interpoint[1].argmin()]))

        first_m = np.array([np.mean(first_line[:,0]), np.mean(first_line[:,1])])
        second_m = np.array([np.mean(second_line[:,0]), np.mean(second_line[:,1])])
            
        # Calculate angle for first point

        third_point = np.array([self._center_x, first_m[1]])

        gegenkathete = distance.cdist([first_m], [third_point])[0][0]
        hypothenuse = distance.cdist([first_m], [[self._center_x, self._center_y]])[0][0]

        rads = np.arcsin(gegenkathete / hypothenuse)
        degs_first = math.degrees(rads)

        # Calculate angle for second point

        third_point = np.array([self._center_x, second_m[1]])

        gegenkathete = distance.cdist([second_m], [third_point])[0][0]
        hypothenuse = distance.cdist([second_m], [[self._center_x, self._center_y]])[0][0]

        g_h = gegenkathete / hypothenuse

        rads = math.asin(g_h)
        degs_second = math.degrees(rads)

        # Determine quadrant we're in to add respective 90 degrees intervals
        # Also as we are actually calculating the reverse angles if we are
        # in the upper left or lower right quadrant of the image
        # we need to take the calculated angle and subtract it from the full 90 degrees
        
        # Also we sanitize for a few special cases here where points are exactly on
        # the centers x or y coordinate

        # First point

        if first_m[0] > self._center_x and first_m[1] > self._center_y:
            degs_first = (90 - degs_first) + 90
        elif first_m[0] < self._center_x and first_m[1] > self._center_y:
            degs_first += 180
        elif first_m[0] < self._center_x and first_m[1] < self._center_y:
            degs_first = (90 - degs_first) + 270
        elif first_m[0] == self._center_x and first_m[1] > self._center_y:
            degs_first = 180
        elif first_m[0] == self._center_x and first_m[1] < self._center_y:
            degs_first = 0
        elif first_m[0] < self._center_x and first_m[1] == self._center_y:
            degs_first = 270
        elif first_m[0] > self._center_x and first_m[1] == self._center_y:
            degs_first = 90

        # Sencond point

        if second_m[0] > self._center_x and second_m[1] > self._center_y:
            degs_second = (90 - degs_second) + 90
        elif second_m[0] < self._center_x and second_m[1] > self._center_y:
            degs_second += 180
        elif second_m[0] < self._center_x and second_m[1] < self._center_y:
            degs_second = (90 - degs_second) + 270
        elif second_m[0] == self._center_x and second_m[1] > self._center_y:
            degs_second = 180
        elif second_m[0] == self._center_x and second_m[1] < self._center_y:
            degs_second = 0
        elif second_m[0] < self._center_x and second_m[1] == self._center_y:
            degs_second = 270
        elif second_m[0] > self._center_x and second_m[1] == self._center_y:
            degs_second = 90
        
        # Sanitize a few special cases

        # For the first point

        if degs_first < degs_second:
            return (degs_first, degs_second)
        else:
            return (degs_second, degs_first)

    def find_all_shape_width(self):
        shape_min = []
        shape_max = []
        for shape in self._shapes.values():
            mini, maxi = self._process_shape(shape)
            shape_min.append(mini)
            shape_max.append(maxi)
        
        # import pdb; pdb.set_trace()
        arr = np.column_stack((list(self._shape_ids), shape_min, shape_max))
        diff = arr[:,2] - arr[:,1]
        diff[diff > 200] = 360 - diff[diff > 200]
        arr = np.c_[arr, diff]
        return arr

class MidiMaker(object):
    """
    ::np.array arr_track_min_max:: array of [track id (Tonspur?), start degree, end_degree] of shape found on track
    ::int duration_ttl:: duration of track in seconds to match degrees to start/end time of pitch
    ::dict tracks_to_note:: map track id to pitch for midi conversion
    ::int tempo:: BPM
    ::int octave:: specify octave of music peace for note - midi conversion
    """
    def __init__(self, arr_track_min_max,
                 tracks_to_note,
                 beats_total=144
                 ):
        self.data_array = arr_track_min_max
        self.tracks_to_note = tracks_to_note
        self.beats = beats_total
        self.degrees_per_beat = 360 / beats_total

    def _convert_track_degree(self):
        start_time = (360 - self.data_array[:,2]) / self.degrees_per_beat
        duration = self.data_array[:,3] / self.degrees_per_beat
        pitch = np.vectorize(self.tracks_to_note.get)(self.data_array[:,0])
        return (start_time, duration, pitch)

    def create_midi(self, out_file):
        start_time, duration, pitch = self._convert_track_degree()
        midi_obj = MIDIFile(numTracks=len(self.tracks_to_note),
                    removeDuplicates=False,  # set True?
                    deinterleave=True,  # default
                    adjust_origin=False,
                    # default - if true find earliest event in all tracts and shift events so that time is 0
                    file_format=1,  # default - set tempo track separately
                    ticks_per_quarternote=480,  # 120, 240, 384, 480, and 960 are common values
                    eventtime_is_ticks=False  # default
                    )

        #for track_id in tpm.tracks_to_note.keys():
        #    midi_obj.addTempo(track_id, time=0, tempo=tpm.tempo)

        channel = 0 # we do not have multiple instruments
        volume = 100
        for i, _ in enumerate(start_time):
            midi_obj.addNote(track = (int(self.data_array[i, 0]) - 1),
                                channel = channel,
                                pitch = pitch[i],
                                time = start_time[i],
                                duration = duration[i],
                                volume = volume)

        with open(out_file, "wb") as output_file:
            midi_obj.writeFile(output_file)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help = "input image file, must be able to open with cv2.imread")
    parser.add_argument("output", help = "output file name")
    parser.add_argument("-s", "--shapes-file", help = "file name to save detected shapes to, defaults to 'detected_shapes.tiff'",
                        const = None, nargs = "?", default = None)
    parser.add_argument("-t", "--tracks-file", help = "file name to save detected tracks to if desired",
                        const = None, nargs = "?", default = None)
    parser.add_argument("-c", "--config", help = "config file containing required information about plate type",
                        const = "config.yaml", default = "config.yaml", nargs = "?")
    parser.add_argument("-d", "--disc-type", help = "type of the plate to process",
                        const = "default", default = "default", nargs = "?")
    args = parser.parse_args()
    
    print("Reading config file from '", args.config, "'... ", sep = "", end = "")
    
    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("{:>10}".format("FAIL"))
            print("Could not read config file, original error:", exc)
            sys.exit()

    print("{:>10}".format("OK"))

    print("Using configuration preset '", args.disc_type, "'... ", sep = "", end = "")

    # config = config["default"]
    config = config[args.disc_type]

    print("{:>10}".format("OK"))

    print("Reading input image from '", args.input, "'... ", sep = "", end = "")

    # Read image in
    # picture = cv2.imread("data/test_rotated.tiff", cv2.IMREAD_GRAYSCALE)
    picture = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

    print("{:>10}".format("OK"))

    print("Finding connected components... ", end = "")


    # Create labels
    processor = image_processor(picture, config["search_distance"])
    labels, labels_image = processor.label()

    print("{:>10}".format("OK"))

    if not args.shapes_file is None:
        print("Writing image of detected shapes to '", args.shapes_file, "'... ", sep = "", end = "")
        cv2.imwrite(args.shapes_file, labels_image)
        print("{:>10}".format("OK"))

    print("Segmenting disc into tracks... ", end = "")

    # shapes_dict, assignments, color_image = processor.extract_shapes(outer_radius, inner_radius, center_x, center_y, 10)
    shapes_dict, assignments, color_image = processor.extract_shapes(config["outer_radius"],
                                                                     config["inner_radius"],
                                                                     config["center_x"],
                                                                     config["center_y"],
                                                                     config["bandwidth"])

    print("{:>10}".format("OK"))

    if not args.tracks_file is None:
        print("Writing image of detected tracks to '", args.tracks_file, "'... ", sep = "", end = "")
        cv2.imwrite(args.tracks_file, color_image)
        print("{:>10}".format("OK"))

    print("Calculating position of detected notes... ", end = "")

    ts = time_segmenter(shapes_dict, shapes_dict.keys(), config["center_x"], config["center_y"])
    arr = ts.find_all_shape_width()
    arr = np.column_stack((arr, assignments[:,1]))

    # Mutate the order to the way our midi writer expects them
    per = [4, 1, 2, 3, 0]
    arr[:] = arr[:,per]

    print("{:>10}".format("OK"))

    print("Creating midi output and writing to '", args.output, "'... ", sep = "", end = "")

    midi_maker = MidiMaker(arr_track_min_max = arr, tracks_to_note = config["track_mappings"])
    # midi_maker.create_midi("miiiidiiii.mid")
    midi_maker.create_midi(args.output)

    print("{:>10}".format("OK"))

    #np.save("data/processed_shapes.npy", arr)
    #np.savetxt("angles.txt", arr, fmt = "%1.3f", delimiter = ",")

if __name__ == "__main__":
    main()