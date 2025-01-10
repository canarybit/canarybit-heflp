from models import FCN
import numpy as np
# from heflp.training.params import save_flattened_model_params
from threatintellidataset import *
from heflp.training.params import flatten_model_params
import keras
import torch
import sys
import pandas as pd
timesteps_max = 10

def timesteps_calculation(df, timesteps_max):

    timesteps_calc = int(pd.Series(df.ProcessGuid).value_counts().quantile(0.5))
    if timesteps_calc >= timesteps_max:
        timesteps = timesteps_max
    else:
        timesteps = timesteps_calc

    print("timesteps:", timesteps)

    return timesteps

# def flatten_model_params(model):
#     '''Flatten the model into a 1D Numpy array (Vector)'''
#     if "keras" in sys.modules and isinstance(model,Model):
#         params = np.concatenate(
#             [param.flatten() for param in model.get_weights()]
#         )
#     else:
#         raise ValueError("Invalid model type. Expecting PyTorch or Keras model.")
#     return params

def unflatten_model_params(flattened_params:NDArray, model):
    '''Unflatten the 1D Vector and update the model parameters accordingly'''
    if "torch" in sys.modules and isinstance(model, torch.nn.Module):
        start = 0
        for param in model.parameters():
            end = start + param.numel()
            param.data = torch.from_numpy(flattened_params[start:end].reshape(param.shape).astype(np.float32))
            start = end
    elif "keras" in sys.modules and isinstance(model, keras.Model):
        unflattened_weights = []
        start = 0
        count = 1

        print("Shape of flattened_params:", len(flattened_params))
        for param in model.get_weights():
            print("Layer", count, "shape:", param.shape)
            count += 1

        for param in model.get_weights():
            shape = param.shape
            print("Model shape:", shape)
            end = start + np.prod(shape)
            unflattened_weight = np.array(flattened_params[start:end]).reshape(shape).astype(np.float32)
            unflattened_weights.append(unflattened_weight)
            start = end
        model.set_weights(unflattened_weights)
    else:
        raise ValueError("Invalid model type. Expecting PyTorch or Keras model.")
    return model

def save_flattened_model_params(filepath:str, model):
    '''Save the model parameters as a 1D vector'''
    params = flatten_model_params(model)
    print('Shape of the param:', len(params))
    np.save(filepath, params)

input_file = sys.argv[1]
X_train = reading_files(input_file)
print("\nTraining data original shape:", X_train.shape)

timesteps = timesteps_calculation(X_train, timesteps_max)
X_train = reshaping_data(X_train, timesteps=timesteps,test=False)
print("\nTraining data shape:", X_train.shape)

lstm_autoencoder = create_autoencoder(X_train.shape, DEFAULT_MODEL_CONF)
lstm_autoencoder.summary()

save_flattened_model_params("lstm_init.npy", lstm_autoencoder)
paras = np.load('lstm_init.npy')
print(len(paras[0]))

unflatten_model_params(paras[0], lstm_autoencoder)
