import numpy as np
import cv2
import os
import musicbox.image.label
import musicbox.image.preprocessing
import musicbox.image.center
import musicbox.image.notes
import musicbox.notes.midi

# This is the main processing class we use in our pipeline
# It abstracts most complexity from the outside.

class Processor(object):
    data = None
    config = None
    labels = None
    center_x = None
    center_y = None
    debug_dir = ""

    def __init__(self, data, config, debug_dir = False):
        self.data = data
        self.config = config
        self.debug_dir = debug_dir
        return None

    def validate_config(config):
        """Validate that a config file has all required values

        Args:
            config (dict): Config to validate

        Raises:
            NotImplementedError: Not implemented yet
        """
        raise NotImplementedError

    @classmethod
    def from_array(cls, data_array: np.ndarray, config: dict, debug_dir = ""):
        """Creates a new processor class from an image given as a numpy array

        Args:
            data_array (np.ndarray): The image to be processed
            config (dict): Config data for the image to be processed (as read from an config file)
            debug (str, optional): Directory to put debug files in. Empty string to not generate debug files. Defaults to "".

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
        return cls(data_array, config, debug_dir)

    @classmethod
    def from_file(cls, path: str, config: dict, debug_dir = ""):
        """Create a new processor class from an image file on disk

        Args:
            path (str): Path to the image file
            config (dict): Config data for the image to be processed (as read from an config file)
            debug (str, optional): Directory to put debug files in. Empty string to not generate debug files. Defaults to "".

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
        return cls(img, config, debug_dir)

    def run_pipeline(self):

        # First of we create our base addition arguments dictionary to store arguments used for multiple
        # functions later on like the debugging directory

        additional_arguments_base = dict()

        if self.debug_dir != "":
            additional_arguments_base["debug_dir"] = self.debug_dir

        # Preprocessing:
        self.run_preprocessing(additional_arguments_base.copy())

        # Labeling:
        self.run_labeling(additional_arguments_base.copy())

        # Center detection
        self.run_center_detection(additional_arguments_base.copy())

        # Assign Tracks and extract notes
        self.run_note_extraction(additional_arguments_base.copy())

        self.create_midi(additional_arguments_base.copy())

        return None

    def run_labeling(self, additional_arguments = dict()):
       
        if self.config["search_distance"] == 1:
            method = "2-connected"
        else:
            method = "n-distance"
            additional_arguments["distance"] = self.config["search_distance"]

        self.labels = musicbox.image.label.label(method, self.data, additional_arguments)


    def run_preprocessing(self, additional_arguments = dict()):
        self.data = musicbox.image.preprocessing.preprocess(self.config["preprocessing"], self.data, additional_arguments)

    def run_center_detection(self, additional_arguments = dict()):
        (y, x) = musicbox.image.center.detect_center(self.config["center_method"], self.data, additional_arguments)
        self.center_x = round(x)
        self.center_y = round(y)


    def run_note_extraction(self, additional_arguments = dict()):
        additional_arguments["inner_radius"] = self.config["inner_radius"]
        additional_arguments["outer_radius"] = self.config["outer_radius"]
        additional_arguments["bandwidth"] = self.config["bandwidth"]
        additional_arguments["track_width"] = self.config["track_width"]
        additional_arguments["first_track"] = self.config["first_track"]
        additional_arguments["use_punchholes"] = self.config["use_punchholes"] if "use_punchholes" in self.config.keys() else False 
        additional_arguments["punchhole_side"] = self.config["punchhole_side"] if "punchhole_side" in self.config.keys() else False 

        shapes_dict, assignments, color_image = musicbox.image.notes.extract_notes(self.labels, self.center_x, self.center_y, additional_arguments, False)

        arr = musicbox.image.notes.convert_notes(shapes_dict, self.center_x, self.center_y)

        arr = np.column_stack((arr, assignments[:,1]))

        per = [4, 1, 2, 3, 0]
        arr[:] = arr[:,per]

        too_high = arr[:, 0] <= self.config["n_tracks"]

        arr = arr[too_high, :]

        self.data_array = arr
    
    def create_midi(self, additional_arguments = dict()):
        additional_arguments["track_mappings"] = self.config["track_mappings"]
        additional_arguments["bars"] = 144
        additional_arguments["filename"] = "ProcessorOutput.mid"
        additional_arguments["bpm"] = 200
        musicbox.notes.midi.create_midi(self.data_array, additional_arguments)
