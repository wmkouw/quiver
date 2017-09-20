from __future__ import absolute_import, division, print_function
import json
from flask.json import jsonify
import numpy as np
import keras.backend as K
from contextlib import contextmanager
from quiver_engine.imagenet_utils import preprocess_input, decode_imagenet_predictions
from os import path

def validate_launch(html_base_dir):
    print('Starting webserver from:', html_base_dir)
    assert path.exists(path.join(html_base_dir, 'quiverboard')), 'Quiverboard must be a ' \
                                                                       'subdirectory of {}'.format(html_base_dir)
    assert path.exists(path.join(html_base_dir, 'quiverboard', 'dist')), 'Dist must be a ' \
                                                                               'subdirectory of quiverboard'
    assert path.exists(
        path.join(html_base_dir, 'quiverboard', 'dist', 'index.html')), 'Index.html missing'

def get_evaluation_context():
    return get_evaluation_context_getter()()

def get_evaluation_context_getter():
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.get_default_graph().as_default

    if K.backend() == 'theano':
        return contextmanager(lambda: (yield))

def get_input_config(model):
    '''
        returns a tuple (inputDimensions, numChannels)
    '''

    return (
        model.get_input_shape_at(0)[2],
        model.get_input_shape_at(0)[1]
    ) if K.image_dim_ordering() == 'th' else (
        #tf ordering
        model.get_input_shape_at(0)[1],
        model.get_input_shape_at(0)[2]
    )

def decode_predictions(preds, classes, top):
    if not classes:
        print("Warning! you didn't pass your own set of classes for the model therefore imagenet classes are used")
        return decode_imagenet_predictions(preds, top)

    if len(preds.shape) != 2 or preds.shape[1] != len(classes):
        raise ValueError('you need to provide same number of classes as model prediction output ' + \
                         'model returns %s predictions, while there are %s classes' % (
                             preds.shape[1], len(classes)))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [("", classes[i], pred[i]) for i in top_indices]
        results.append(result)
    return results

def load_sig(input_path):
    return np.load(input_path)

def get_jsonable_obj(obj):
    return json.loads(get_json(obj))

def get_json(obj):
    return json.dumps(obj, default=get_json_type)

def safe_jsonify(obj):
    return jsonify(get_jsonable_obj(obj))

def get_json_type(obj):

    # if obj is any numpy type
    if type(obj).__module__ == np.__name__:
        return obj.item()

    # if obj is a python 'type'
    if type(obj).__name__ == type.__name__:
        return obj.__name__

    raise TypeError('Not JSON Serializable')
