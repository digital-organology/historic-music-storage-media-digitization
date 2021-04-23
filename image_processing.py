#!/usr/bin/env python
import cv2
import argparse
import yaml
import sys
import os
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
    parser.add_argument("-c", "--config", help = "config file containing required information about plate type",
                        const = "config.yaml", default = "config.yaml", nargs = "?")
    parser.add_argument("-t", "--type", help = "type of the plate to process",
                        const = "default", default = "default", nargs = "?")
    parser.add_argument("-d", "--debug-dir", default = None, help = "If specified write shape / track colored files\
                                                                     and number annotation to output directory ")
    parser.add_argument("-s", "--skip-canny", dest = "canny", action = "store_false")
    parser.add_argument("-i", "--img-preprocessing", dest = "prepro", action = "store_true",
                        help = "Perform different image processing steps like erosion.")
    args = parser.parse_args()

    # Additional args definition if debug_dir was specified
    if args.debug_dir:
        if not os.path.exists(args.debug_dir):
            print("Creating folder to store debug files...")
            os.mkdir(args.debug_dir)
        args.shapes_file = os.path.join(args.debug_dir, "shapes.jpg")
        args.tracks_file = os.path.join(args.debug_dir, "tracks.jpg")
    else:
        args.shapes_file = None
        args.tracks_file = None

    
    print("Reading config file from '", args.config, "'... ", sep = "", end = "")
    
    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("{:>10}".format("FAIL"))
            print("Could not read config file, original error:", exc)
            sys.exit()

    print("{:>10}".format("OK"))

    print("Using configuration preset '", args.type, "'... ", sep = "", end = "")

    # config = config["default"]
    config = config[args.type]

    print("{:>10}".format("OK"))

    print("Reading input image from '", args.input, "'... ", sep = "", end = "")

    # Read image in
    # picture = cv2.imread("data/test_rotated.tiff", cv2.IMREAD_GRAYSCALE)
    orig_picture = cv2.imread(args.input)

    if args.prepro:
        print("\nPerforming automatic image preprocessing...", sep = "", end = "")
        processed_picture = change_contrast_brightness(orig_picture,
                                                        config["erosion_kernel"],
                                                        config["noise_kernel"],
                                                        contrast_factor = 1,
                                                        brightness_val = config["brightness"])#1.2
    else:
        print("\nConverting image into grey scale - no preprocessing... ", sep = "", end = "")
        # processed_picture = cv2.cvtColor(orig_picture, cv2.COLOR_BGR2GRAY)

    print("{:>10}".format("OK"))

    if args.canny:
        print("Applying canny algorithm and finding center... ", sep = "", end = "")
        canny_image = musicbox.image.canny.canny_threshold(orig_picture, processed_picture, config["canny_low"], config["canny_high"])
    else:
        print("Finding center... ", sep = "", end = "")
        # This should be done somewhere else and is here just for testing
        _, canny_image = cv2.threshold(orig_picture, 60, 255, cv2.THRESH_BINARY)#original picture or processed picture??

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
                                                                                absolute_mode = False,
                                                                                debug_dir = args.debug_dir,
                                                                                use_punchhole = config["punchholes"],
                                                                                punchhole_side = "left")


    print("{:>10}".format("OK"))

    if args.tracks_file:
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