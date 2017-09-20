import re
import numpy as np
from os.path import relpath, abspath
from os import listdir

def save_layer_sig(layer_outputs, layer_name, idx, temp_folder, input_path):

    # Find out where to write network layer outputs
    filename = get_output_filename(layer_name, idx, temp_folder, input_path)

    # Save network layer outputs as numpy arrays
    np.save(filename, layer_outputs)

    # Return path to layer output arrays
    return relpath(filename, abspath(temp_folder))

def get_output_filename(layer_name, z_idx, temp_folder, input_path):
    return '{}/{}_{}_{}.npy'.format(temp_folder, layer_name, str(z_idx), input_path)

def list_sig_files(input_folder):
    image_regex = re.compile(r'.*\.(npy)$')
    return [
        filename
        for filename in listdir(
            abspath(input_folder)
        )
        if image_regex.match(filename) is not None
    ]
