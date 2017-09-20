import re
import numpy as np
import matplotlib as plt
from os.path import relpath, abspath
from os import listdir

from quiver_engine.timeseries_visualization import plot_timeseries

def save_layer_sig(layer_outputs, layer_name, idx, temp_folder, input_path):

    # Make visualization of layer ouput
    time_plot = plot_timeseries(layer_outputs,dt=0.01)
    time_plot['fig'].save_fig(get_output_png_filename(layer_name,
                idx,
                temp_folder,
                input_path))

    # Save network layer outputs as numpy arrays
    np.save(get_output_npy_filename(layer_name,
                idx,
                temp_folder,
                input_path),
            layer_outputs)

    # Return path to layer output arrays
    return relpath(filename, abspath(temp_folder))

def get_output_npy_filename(layer_name, z_idx, temp_folder, input_path):
    return '{}/{}_{}_{}.npy'.format(temp_folder, layer_name, str(z_idx), input_path[:-4])

def get_output_png_filename(layer_name, z_idx, temp_folder, input_path):
    return '{}/{}_{}_{}.png'.format(temp_folder, layer_name, str(z_idx), input_path[:-4])

def list_sig_files(input_folder):
    image_regex = re.compile(r'.*\.(npy)$')
    return [
        filename
        for filename in listdir(
            abspath(input_folder)
        )
        if image_regex.match(filename) is not None
    ]
