import re
import numpy as np
# import matplotlib as plt
from os.path import relpath, abspath
from os import listdir

from quiver_engine.timeseries_visualization import plot_timeseries

def save_layer_sig(layer_outputs, layer_name, temp_folder, input_path):

    # Generate filenames
    fn_npy = get_output_filename(layer_name, temp_folder, input_path, ext='npy')
    fn_png = get_output_filename(layer_name, temp_folder, input_path, ext='png')

    # Make visualization of layer ouput
    time_plot = plot_timeseries(np.atleast_2d(layer_outputs),dt=0.01)
    time_plot['fig'].savefig(fn_png)

    # Save network layer outputs as numpy arrays
    np.save(fn_npy, np.atleast_2d(layer_outputs))

    # Return path to layer output arrays
    return relpath(fn_png, abspath(temp_folder)), relpath(fn_npy, abspath(temp_folder))

def get_output_filename(layer_name, temp_folder, input_path, ext='png'):
    return '{}/{}_{}.{}'.format(temp_folder, layer_name, input_path[:-4], ext)

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
