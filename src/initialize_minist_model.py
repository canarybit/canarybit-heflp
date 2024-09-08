from models import FCN
import numpy as np
from heflp.training.params import save_flattened_model_params

model = FCN()
save_flattened_model_params("fcn_init.npy", model)
array = np.load('fcn_init.npy')
print(array.shape)