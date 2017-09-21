import re
import numpy as np
# import matplotlib as plt
from os.path import relpath, abspath, isfile
from os import listdir

import quiver_engine.timeseries_visualization as tsv

def save_layer_sig(layer_outputs, layer_name, kernel, temp_folder, input_path):

    # Generate filenames
    fn_png = get_output_filename(layer_name, temp_folder, input_path, str(kernel), ext='png')

    if not isfile(fn_png):
        # Plot and write timeseries per kernel
        tsv.generate_timeseries_image(np.atleast_2d(layer_outputs), fn_png)

    # Return path to layer output arrays
    return relpath(fn_png, abspath(temp_folder))

def get_output_filename(layer_name, temp_folder, input_path, kernel_index='', ext='png'):
    return '{}/{}_{}_{}.{}'.format(temp_folder, layer_name, input_path[:-4], kernel_index, ext)

def list_sig_npy_files(input_folder):
    image_regex = re.compile(r'.*\.(npy)$')
    return [
        filename
        for filename in listdir(
            abspath(input_folder)
        )
        if image_regex.match(filename) is not None
    ]

def list_sig_png_files(input_folder):
    image_regex = re.compile(r'.*\.(png)$')
    return [
        filename
        for filename in listdir(
            abspath(input_folder)
        )
        if image_regex.match(filename) is not None
    ]
