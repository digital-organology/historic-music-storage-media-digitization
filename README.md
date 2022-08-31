# Music box disc digitization

This repository contains code to automatically generate a midi file out of a high resolution image of historic musical storage media.
Currently cardboard discs can be processed, configuration files for Ariston type discs as well as ones for an *mechanischer Klaviervorsetzer* are included.
Processing of piano rolls is currently under development with an experimental pipeline providing support for the Hupfeld Phonola Solodant type of rolls.
The pipeline is highly modular and is continuously expanded to include processing for a number of types of historic music storage media and additional analytic features.
This is an in-progress project at the Digital Organology group of the Museum of Musical Instruments at Leipzig University.

## Prerequisites

A working installation of python 3 (will most likely not work with python 2, we did not test that) is required.
Additionally the packages in `requirements.txt` need to be installed.
This can be done using PIP:

```{bash}
pip install -r requirements.txt
```

A proper installation routine with `Setuptools` will be provided in the future.

## Image preprocessing

Currently the script is developed using backlit images of round music box discs like this:

![example_disc](./images/example.JPG)

Depending on the way your images are taken there might be additional preprocessing required, images similar to the one above should work out of the box.

## Usage

Pipelines are configured in the `config.yaml` file.
Some example pipelines are already included.

The software can be accessed either directly in python our through our wrapper script.

### Python usage

We provide the `processor` class. It can be used to process a file. An example can be seen in our wrapper file:

```{python}
with open(args.config, "r") as stream:
    config = yaml.safe_load(stream)

config = config["ariston"]

disc_processor = musicbox.processor.Processor.from_file("image_of_an_ariston_disc.JPG", config, debug_dir = "put_debug_files_here", verbose = True)

disc_processor.run()
```

### Wrapper script

Alternatively, use the `process.py` wrapper:

```{bash}
python process.py -d "put_debug_files_here" -v -t ariston image_of_an_ariston_disc.JPG
```

All available arguments can be seen with the `--help` switch.

## Contributing

Adding new pipeline components is done by adding your method to the `processor_config.yaml`.
Take an example of the existing methods to manage what your component needs and provides.
Note that your method is not limited to accessing or storing the information provided there, this information is only used to validate if the pipeline is runnable.

Your methods name is specified by where it lives and the function name itself.
A function called `crop_image` that is located in `preprocessing.py` should be called `preprocessing.crop_image` to be found by the dispatcher.

Your method will receive one argument which is a reference to the `processor` instance that invoked it.
It may read any information it needs from that instance and write data that is supposed to be available to other methods to it (take care not to overwrite anything, best check what is defined to None in the class and add your slots).
Your method should return `True` if everything went okay and `False` elsewise, though we might drop that requirement in the future.

After this is done, you can include your method in a pipeline in `config.yaml`.
