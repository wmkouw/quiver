import numpy as np
import keras.backend as K
from keras.models import Model
from quiver_engine.util import get_evaluation_context
from quiver_engine.file_utils import save_layer_sig
from quiver_engine.layer_result_generators import get_outputs_generator

def save_layer_outputs(input_sig, model, layer_name, temp_folder, input_path):

    with get_evaluation_context():
        inter_model = Model(input=model.input,output=model.get_layer(layer_name).output)
        layer_outputs = np.atleast_3d(inter_model.predict(input_sig, verbose=0)[0])

        if K.backend() == 'theano':
            #correct for channel location difference betwen TF and Theano
            layer_outputs = np.rollaxis(layer_outputs, 0, 2)

        return [
            save_layer_sig(
                layer_outputs[:,:,kernel],
                layer_name,
                kernel,
                temp_folder,
                input_path
            )
            for kernel in range(0, layer_outputs.shape[2])
        ]
