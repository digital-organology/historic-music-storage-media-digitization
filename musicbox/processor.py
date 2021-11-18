import numpy as np
import cv2

# This is the main processing class we use in our pipeline
# It abstracts most complexity from the outside.

class processor(object):

    data = None
    config = None

    def __init__(self, data, config):
        self.data = data
        self.config = config
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
    def from_array(cls, data_array: np.ndarray, config: dict):
        """Creates a new processor class from an image given as a numpy array

        Args:
            data_array (np.ndarray): The image to be processed
            config (dict): Config data for the image to be processed (as read from an config file)

        Raises:
            TypeError: If either argument is of a bad type this will be raised  

        Returns:
            processor: An instance of the processor class containing the given parameters
        """
        if not isinstance(data_array, np.ndarray):
            raise TypeError("data_array must be a ndarray")
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        return cls(data_array, config)

    @classmethod
    def from_file(cls, path: str, config: dict):
        """Create a new processor class from an image file on disk

        Args:
            path (str): Path to the image file
            config (dict): Config data for the image to be processed (as read from an config file)

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
        img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
        if img == None or not isinstance(img, np.ndarray):
            raise ValueError("Image could not be read from provided path")
        return cls(img, config)

    def run_pipeline():
        
            a = kekse * 2
        return None



    def lol():

