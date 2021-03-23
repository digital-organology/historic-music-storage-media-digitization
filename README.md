# Music box disc digitization

This repository contains code to automatically generate a midi file out of a high resolution image of discs for old music boxes.
This is an in-progress project at the Digital Organology group of the Museum of Musical Instruments at Leipzig University.

## Prerequisites

A working installation of python 3 (may also work with python 2, we did not test that) is required.
Additionally the packages in `requirements.txt` need to be installed.
This can be done using PIP:

```{bash}
pip install -r requirements.txt
```

## Image preprocessing

Currently the script is developed using backlit images of round music box discs like this:

![example_disc](./images/example.JPG)

Depending on the way these images are taken some preprocessing might be required.
We currently decrease the brightness and increase the contrast.
This can be done using ImageMagick:

```{bash}
convert image.JPG -brightness-contrast -60x40 image_out.JPG
```

The script might also work with non-backlit pictures (and was in fact using these in the beginning of development) though some parameter tuning (see `config.yaml`) might be required and results may be of lesser quality.

## Usage

The main script can be run on the console via

```{bash}
python image_processing.py [input_image] [output_file]
```

There are a number of parameters:

* `-h` Outputs information about parameters
* `-s` Specifies where an image of the detected connected components will be written to if desired
* `-t` The same as above but for the output of the track detection algorithm
* `-c` Path to the config file, may be omitted if the file is called `config.yaml`
* `-d` The type of the disc, used for selecting a configuration preset
