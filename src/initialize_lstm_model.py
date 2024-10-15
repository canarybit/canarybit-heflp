from models import FCN
import numpy as np
from heflp.training.params import save_flattened_model_params

initial_seq = Sequential()
save_flattened_model_params("fcn_init.npy", initial_seq)
array = np.load('lstm_init.npy')
print(array.shape)