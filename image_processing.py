#!/usr/bin/env python
import cv2
import argparse
import yaml
import sys
import numpy as np
import musicbox.image.label
import musicbox.image.canny
import musicbox.image.center
import musicbox.image.notes
import musicbox.notes.convert
import musicbox.notes.midi
from musicbox.image.processing import change_contrast_brightness

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
    orig_picture = cv2.imread(args.input)
    picture = change_contrast_brightness(orig_picture, contrast_factor=1, brightness_val=100)#1.2

    print("{:>10}".format("OK"))

    print("Applying canny algorithm and finding center... ", sep = "", end = "")

    canny_image = musicbox.image.canny.canny_threshold(orig_picture, picture, config["canny_low"], config["canny_high"])

    center_x, center_y = musicbox.image.center.calculate_center(canny_image)

    img_grayscale = cv2.cvtColor(canny_image, cv2.COLOR_BGR2GRAY)

    print("{:>10}".format("OK"))

    print("Finding connected components... ", end = "")

    # Create labels
    labels, labels_image = musicbox.image.label.label_image(img_grayscale, config["search_distance"])

    print("{:>10}".format("OK"))

    if args.shapes_file:
        print("Writing image of detected shapes to '", args.shapes_file, "'... ", sep = "", end = "")
        cv2.imwrite(args.shapes_file, labels_image)
        print("{:>10}".format("OK"))

    print("Segmenting disc into tracks... ", end = "")

    # shapes_dict, assignments, color_image = processor.extract_shapes(outer_radius, inner_radius, center_x, center_y, 10)
    shapes_dict, assignments, color_image = musicbox.image.notes.extract_notes(labels,
                                                                                config["outer_radius"],
                                                                                config["inner_radius"],
                                                                                center_x,
                                                                                center_y,
                                                                                config["bandwidth"],
                                                                                config["first_track"], 
                                                                                config["track_width"],
                                                                                img_grayscale,
                                                                                compat_mode = False,
                                                                                exact_mode = False)

    print("{:>10}".format("OK"))

    if not args.tracks_file is None:
        print("Writing image of detected tracks to '", args.tracks_file, "'... ", sep = "", end = "")
        cv2.imwrite(args.tracks_file, color_image)
        print("{:>10}".format("OK"))

    print("Calculating position of detected notes... ", end = "")

    arr = musicbox.notes.convert.convert_notes(shapes_dict.values(), shapes_dict.keys(), center_x, center_y)

    arr = np.column_stack((arr, assignments[:,1]))


    # Mutate the order to the way our midi writer expects them
    per = [4, 1, 2, 3, 0]
    arr[:] = arr[:,per]

    too_high = arr[:, 0] <= config["n_tracks"]

    arr = arr[too_high, :]

    # np.savetxt("arr.txt", arr, fmt = "%1.3f", delimiter = ",")


    print("{:>10}".format("OK"))

    print("Creating midi output and writing to '", args.output, "'... ", sep = "", end = "")

    musicbox.notes.midi.create_midi(arr, config["track_mappings"], 144, args.output)

    print("{:>10}".format("OK"))

    #np.save("data/processed_shapes.npy", arr)
    

if __name__ == "__main__":
    main()