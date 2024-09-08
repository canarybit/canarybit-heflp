from models import FCN
from heflp.training.params import save_flattened_model_params

model = FCN()
save_flattened_model_params("fcn_init.npy", model)