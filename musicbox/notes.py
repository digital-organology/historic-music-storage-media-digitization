from turtle import back
import numpy as np
import cv2
import math
import timeit
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance
from datetime import datetime
from sklearn.cluster import KMeans
from midiutil.MidiFile import MIDIFile
from musicbox.helpers import gen_lut, make_color_image


def find_beat_length(proc):
    note_length = proc.note_data[:,3]
    note_length = np.column_stack((note_length, np.zeros(len(note_length))))

    fit = KMeans(4)
    fit.fit(note_length)

    if "debug_dir" in proc.parameters:
        start_time = timeit.default_timer()

        plt.scatter(note_length[:,0], note_length[:,1], c = fit.labels_)
        plt.savefig(os.path.join(proc.parameters["debug_dir"], "clustering_beat_length.tiff"))
        
        print("INFO: Creating debug information added an overhead of " + ("%.5f" % (timeit.default_timer() - start_time)) + " seconds")

    # Our assumption here is that the smallest cluster is 1/8 notes, the second smallest thus beeing 1/4 notes
    # So whatever the length of one of those is, is the conversion factor degree/beats

    proc.beat_length = np.sort(fit.cluster_centers_[:,0])[1]

    return True
    


def _calculate_angles(shape, center_x, center_y, return_points = False):
    # This is most likely pretty inefficient

    # As numpy stores things as y,x pairs we need to switch the coordinates around
    # to make sense for opencv

    shape[:,[0, 1]] = shape[:, [1, 0]]
    rectangle = cv2.minAreaRect(shape.astype(np.float32))
    box = cv2.boxPoints(rectangle)
    box = box.astype(np.uint32)

    if len(box) != 4:
        print(len(box))
        return (0, 0)

    # Scipy spatial on the other hand does expect things to be y,x
    # but if we give both coordinates in the wrong order it _should_ be fine
    dists = distance.cdist(box, [[center_x, center_y]])

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

    third_point = np.array([center_x, first_m[1]])

    gegenkathete = distance.cdist([first_m], [third_point])[0][0]
    hypothenuse = distance.cdist([first_m], [[center_x, center_y]])[0][0]

    rads = np.arcsin(gegenkathete / hypothenuse)
    degs_first = math.degrees(rads)

    # Calculate angle for second point

    third_point = np.array([center_x, second_m[1]])

    gegenkathete = distance.cdist([second_m], [third_point])[0][0]
    hypothenuse = distance.cdist([second_m], [[center_x, center_y]])[0][0]

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

    if first_m[0] > center_x and first_m[1] > center_y:
        degs_first = (90 - degs_first) + 90
    elif first_m[0] < center_x and first_m[1] > center_y:
        degs_first += 180
    elif first_m[0] < center_x and first_m[1] < center_y:
        degs_first = (90 - degs_first) + 270
    elif first_m[0] == center_x and first_m[1] > center_y:
        degs_first = 180
    elif first_m[0] == center_x and first_m[1] < center_y:
        degs_first = 0
    elif first_m[0] < center_x and first_m[1] == center_y:
        degs_first = 270
    elif first_m[0] > center_x and first_m[1] == center_y:
        degs_first = 90

    # Sencond point

    if second_m[0] > center_x and second_m[1] > center_y:
        degs_second = (90 - degs_second) + 90
    elif second_m[0] < center_x and second_m[1] > center_y:
        degs_second += 180
    elif second_m[0] < center_x and second_m[1] < center_y:
        degs_second = (90 - degs_second) + 270
    elif second_m[0] == center_x and second_m[1] > center_y:
        degs_second = 180
    elif second_m[0] == center_x and second_m[1] < center_y:
        degs_second = 0
    elif second_m[0] < center_x and second_m[1] == center_y:
        degs_second = 270
    elif second_m[0] > center_x and second_m[1] == center_y:
        degs_second = 90
    # Sanitize a few special cases

    # For the first point

    if degs_first < degs_second:
        if return_points:
            return (first_m, second_m)
        else:
            return (degs_first, degs_second)
    else:
        if return_points:
            return (second_m, first_m)
        else:
            return (degs_second, degs_first)

def create_notes(proc):
    shapes = proc.shapes.values()
    shape_ids = list(proc.shapes.keys())
    shape_min = []
    shape_max = []
    for shape in shapes:
        mini, maxi = _calculate_angles(shape, proc.center_x, proc.center_y)
        shape_min.append(mini)
        shape_max.append(maxi)

    arr = np.column_stack((list(shape_ids), shape_min, shape_max))
    diff = arr[:,2] - arr[:,1]
    # diff[diff > 200] = 360 - diff[diff > 200]
    arr = np.c_[arr, diff]

    if "debug_dir" in proc.parameters:
        start_time = timeit.default_timer()


        background = make_color_image(proc.labels.copy())

        for shape_id, shape in proc.shapes.items():
            # Get the 
            point_1, point_2 = _calculate_angles(shape, proc.center_x, proc.center_y, return_points = True)
            # import pdb; pdb.set_trace()
            background[int(point_1[0]), int(point_1[1])] = (255, 255, 255)
            background[int(point_2[0]), int(point_2[1])] = (255, 255, 255)

            center = np.mean(shape, 0).astype(np.uint32)
            point = (center[1], center[0])
            cv2.putText(background, str(shape_id), point,
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)

        cv2.imwrite(os.path.join(proc.parameters["debug_dir"], "note_conversion.tiff"), background)

        np.savetxt(os.path.join(proc.parameters["debug_dir"], "detected_angles.csv"), arr, delimiter = ",", header = "id, min_angle, max_angle, diff", comments = "")


        print("INFO: Creating debug information added an overhead of " + ("%.5f" % (timeit.default_timer() - start_time)) + " seconds")


    proc.note_data = arr
    return True

def _convert_track_degree(data_array, tracks_to_notes, proc, debug_dir = ""):
    start_time = (360 - data_array[:,3])
    duration = data_array[:,4]

    # import pdb; pdb.set_trace()

    if hasattr(proc, "beat_length") and proc.beat_length is not None:
        start_time = start_time / proc.beat_length
        duration = duration / proc.beat_length

    pitch = data_array[:,0]
    keys = np.array(list(tracks_to_notes.keys()))
    values = np.array(list(tracks_to_notes.values()))

    sidx = keys.argsort()

    ks = keys[sidx]
    vs = values[sidx]

    pitch = vs[np.searchsorted(ks, pitch)]

    arr = np.array((data_array[:,1], start_time, duration, pitch)).T
    proc.data_array = arr

    # pitch = np.vectorize(tracks_to_notes.get)(data_array[:,0])
    if "debug_dir" != "":
        np.savetxt(os.path.join(debug_dir, "music_data.csv"), arr, delimiter = ",", header = "note_id, start, duration, midi_tone", comments = "")
    # import pdb; pdb.set_trace()
    return (start_time, duration, pitch)

def create_midi(proc):
    data_array = np.c_[list(proc.assignments.values()), proc.note_data]

    # At this point data_array contains track, shape_id, min_angle (end), max_angle (start), diff (length)
    start_time, duration, pitch = _convert_track_degree(data_array, proc.parameters["track_mappings"], proc, proc.parameters["debug_dir"] if "debug_dir" in proc.parameters.keys() else "")
    midi_obj = MIDIFile(numTracks=1,
                removeDuplicates=False,  # set True?
                deinterleave=True,  # default
                adjust_origin=False,
                # default - if true find earliest event in all tracts and shift events so that time is 0
                file_format=1,  # default - set tempo track separately
                ticks_per_quarternote=480,  # 120, 240, 384, 480, and 960 are common values
                eventtime_is_ticks=False  # default
                )

    midi_obj.addTempo(0, 0, proc.parameters["bpm"])

    midi_obj.addTimeSignature(0, 0, 4, 2, 24)

    #for track_id in tpm.tracks_to_note.keys():
    #    midi_obj.addTempo(track_id, time=0, tempo=tpm.tempo)

    channel = 0 # we do not have multiple instruments
    volume = 100
    for i, _ in enumerate(start_time):
        midi_obj.addNote(track = 0,
                         channel = channel,
                         pitch = pitch[i],
                         time = start_time[i],
                         duration = duration[i],
                         volume = volume)

    filename = "output_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".mid"

    proc.midi_filename = filename

    if proc.verbose:
        print("INFO: Writing midi file to '" + filename + "'...")

    with open(filename, "wb") as output_file:
        midi_obj.writeFile(output_file)