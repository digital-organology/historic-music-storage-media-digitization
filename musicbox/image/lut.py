import numpy as np

def gen_lut():
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