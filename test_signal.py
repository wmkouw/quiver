# Test quiver-time
import keras as ks
from quiver_engine import server

# Load model
model = ks.models.load_model('models/my_bestmodel.h5')

# Launch server
server.launch(model, classes=range(6), input_folder='input/')
