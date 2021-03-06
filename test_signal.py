# Test quiver-time
import keras as ks
from quiver_engine import server
import sys

# Load model
if sys.version_info.major == 2:
    model = ks.models.load_model('models/my_bestmodel.h5')
elif sys.version_info.major == 3:
    model = ks.models.load_model('models/my_bestmodel_py3.h5')

# Names of classes of PAMAP2
classes = ['lying',
    'sitting',
    'standing',
    'walking',
    'cycling',
    'vacccuum',
    'ironing']

# Launch server
server.launch(model, classes=classes, top=7, input_folder='input/')
