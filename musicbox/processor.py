import numpy as np
import cv2
import os
import yaml
import sys
import timeit
from pathlib import Path
# We could dynamically import these with __import__ or importlib if we want to
import musicbox.image
import musicbox.preprocessing
import musicbox.notes
import musicbox.shapes
import musicbox.tracks
import musicbox.analysis

# This is the main processing class

class Processor(object):

    # These attributes hold data required for the actual data processing
    # We do not actually need to declare them here as they can be added dynamically
    # but we do this to create a better overview of what exists
    original_image = None
    current_image = None
    # This contains shape_id, min_angle (end), max_angle (start), diff (length)
    note_data = None
    # This has note_id, start_time, duration, pitch
    data_array = None
    labels = None
    shapes = None
    center_x = None
    center_y = None
    disc_edge = None
    label_edge = None
    note_data = None
    midi_filename = None
    assignments = None
    track_distances = None
    beat_length = None

    # These are configuration variables
    pipeline = None
    parameters = None
    verbose = False
    debug_dir = ""

    def __init__(self, data, config, debug_dir = "", verbose = False):
        self.original_image = data.copy()
        self.current_image = data.copy()
        if not {"pipeline", "config"} <= config.keys():
            raise ValueError("config provided is invalid")
        self.parameters = config["config"]
        if not debug_dir == "":
            self.parameters["debug_dir"] = debug_dir
        self.pipeline = config["pipeline"]
        self.parsed_config = None
        self.debug_dir = debug_dir
        self.verbose = verbose
        return None

    @classmethod
    def from_array(cls, data_array: np.ndarray, config: dict, debug_dir = "", verbose = False):
        """Creates a new processor class from an image given as a numpy array

        Args:
            data_array (np.ndarray): The image to be processed
            config (dict): Config data for the image to be processed (as read from an config file)
            debug (str, optional): Directory to put debug files in. Empty string to not generate debug files. Defaults to "".
            verbose (bool): Should the current status be output to the standard console

        Raises:
            TypeError: If either argument is of a bad type this will be raised  

        Returns:
            processor: An instance of the processor class containing the given parameters
        """
        if not isinstance(data_array, np.ndarray):
            raise TypeError("data_array must be a ndarray")
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        if debug_dir != "" and not os.path.exists(debug_dir):
            os.mkdir(debug_dir)
        return cls(data_array, config, debug_dir, verbose)

    @classmethod
    def from_file(cls, path: str, config: dict, debug_dir = "", verbose = False):
        """Create a new processor class from an image file on disk

        Args:
            path (str): Path to the image file
            config (dict): Config data for the image to be processed (as read from an config file)
            debug (str, optional): Directory to put debug files in. Empty string to not generate debug files. Defaults to "".
            verbose (bool): Should the current status be output to the standard console

        Raises:
            TypeError: If either argument is of a bad type this will be raised
            ValueError: Will be raised if path does not point towards a readable image file

        Returns:
            processor: An instance of the processor class containing the given parameters
        """
        if not isinstance(path, str):
            raise TypeError("data_array must be a ndarray")
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # import pdb; pdb.set_trace()
        if not isinstance(img, np.ndarray):
            raise ValueError("Image could not be read from provided path")
        if debug_dir != "" and not os.path.exists(debug_dir):
            os.mkdir(debug_dir)
        return cls(img, config, debug_dir, verbose)

    def prepare_pipeline(self):
        p_config_path = os.path.join(Path(__file__).parent.absolute(), "processor_config.yaml")
        with open(p_config_path, "r") as stream:
            try:
                p_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError("Error occured while reading processor_config.yaml file. Original error: " + exc)

        method_config = p_config["methods"]

        if self.verbose:
            print("Validating and preparing pipeline...")

        # Add debug dir to parameters if we have it

        params = dict()

        if self.debug_dir != "":
            params["debug_dir"] = self.debug_dir

        # Validate each pipeline step and add required parameters to the dictd

        available_data = []
        is_runnable = True
        errors = []
        warnings = []

        for pipeline_step in self.pipeline:
            if not pipeline_step in method_config.keys():
                warnings.append("Could not find config for pipeline step '" + pipeline_step + "'. Will continue but no pipeline validation or parameter detection will be available. Make sure to add an entry to 'processor_config.yaml'.")
                continue

            step_config = method_config[pipeline_step]

            # Check if we have all the date required for this 
            if not set(step_config["requires"]) <= set(available_data):
                missing_steps = ", ".join(set(step_config["requires"]) - set(available_data))
                errors.append("Pipeline step '" + pipeline_step + "' requires components '" + missing_steps + "' to be computed but the given pipeline does not provide them")
                is_runnable = False

            if not set(step_config["suggests"]) <= set(available_data):
                missing_steps = ", ".join(set(step_config["suggests"]) - set(available_data))
                warnings.append("Pipeline step '" + pipeline_step + "' suggests components '" + missing_steps + "' to be computed but the given pipeline does not provide them")

            available_data.extend(step_config["provides"])

            for parameter in step_config["parameters"]:
                if not parameter in self.parameters.keys():
                    errors.append("Pipeline step '" + pipeline_step + "' requires parameter '" + parameter + "' to be set but it is missing in the config file")
                    is_runnable = False

        if not is_runnable:
            print("Could not configure pipeline. The following errors occured:")
            print(*errors, sep = "\n")
            return False

        if warnings:
            print("The following warnings occured:")
            print(*warnings, sep = "\n")

        if self.verbose:
            print("Pipeline checked and executable")

        return True

    def execute_pipeline(self):
        for step in self.pipeline:
            if self.verbose:
                print("Executing pipeline step '" + step + "'...")
                start_time = timeit.default_timer()

            success = self.execute_method(step)

            if success is None:
                print("Warning: Step '" + step + "' finished with an unknown status. Took " + ("%.5f" % (timeit.default_timer() - start_time)) + " seconds to finish")
                continue

            if not success:
                print("Error when executing pipeline step '" + step + "'. Giving up.")
                return False

            if self.verbose:
                print("OK, took " + ("%.5f" % (timeit.default_timer() - start_time)) + " seconds to finish")

        return True

    def execute_method(self, method):
        (module, method) = method.split(".", 1)
        module_name = "musicbox." + module
        module = sys.modules.get(module_name)

        if module is None:
            print("Error: Could not find module '" + module_name + "'. Aborting.")
            return False

        if not hasattr(module, method):
            print("Error: Module '" + module_name + "' does not contain method '" + method + "'. Aborting.")
            return False

        method_to_call = getattr(module, method)

        method_to_call(self)

        return True

    def run(self):
        if not self.prepare_pipeline():
            print("Error when configuring pipeline, giving up.")
            return False

        if not self.execute_pipeline():
            print("Error when executing pipeline, giving up.")
            return False