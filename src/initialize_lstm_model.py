from models import FCN
import numpy as np
# from heflp.training.params import save_flattened_model_params
from threatintellidataset import *

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

def flatten_model_params(model):
    '''Flatten the model into a 1D Numpy array (Vector)'''
    if "keras" in sys.modules and isinstance(model,Model):
        params = np.concatenate(
            [param.flatten() for param in model.get_weights()]
        )
    else:
        raise ValueError("Invalid model type. Expecting PyTorch or Keras model.")
    return params

def save_flattened_model_params(filepath:str, model):
    '''Save the model parameters as a 1D vector'''
    np.save(filepath, flatten_model_params(model))

input_file = sys.argv[1]

X_train = pd.read_csv(input_file)
if 'Unnamed: 0' in X_train.columns:
    X_train.drop(columns=['Unnamed: 0'], inplace=True)
print("\nTraining data original shape:", X_train.shape)

timesteps = timesteps_calculation(X_train, timesteps_max)
X_train = reshaping_data(X_train, timesteps=timesteps,test=False)
print("\nTraining data shape:", X_train.shape)

lstm_autoencoder = create_autoencoder(X_train.shape, DEFAULT_MODEL_CONF)
lstm_autoencoder.summary()

save_flattened_model_params("lstm_init.npy", lstm_autoencoder)
array = np.load('lstm_init.npy')
print(array.shape)
