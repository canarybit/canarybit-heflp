from models import FCN
import numpy as np
from heflp.training.params import save_flattened_model_params
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Masking, Dense, Dropout, LSTM, RepeatVector, TimeDistributed, Bidirectional

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

def reshaping_data(X, timesteps, test):
    
    """
    Reshapes the input data into a suitable format for LSTM models.
    
    Args:
        X (ndarray): Input data array of shape (n_samples, features).
        timesteps (int): Number of time steps or sequence length for the reshaped data.
        test (bool): Specify whether to reshape the tet or training dataset. If test=True it will be reshaped, otherwise train.
        
    Returns:

        ndarray: Reshaped input data array of shape (n_samples - timesteps + 1, timesteps, features).
    
    """
    
    if test:
        Xs = pd.DataFrame()
        for i in X.ProcessGuid.unique():
            sorteddf=X[X.ProcessGuid == i].sort_values(by="UtcTime")
            temp_col = sorteddf["UtcTime"]
            Attack = sorteddf["Attack"]
            matrix_temporal = sorteddf.drop(["ProcessGuid","UtcTime","Attack","collector_node_id"], axis=1).values # "collector-node-id"

            # Define padding values and amount
            padding_value = -1  # Change this to the value you want for padding
            padding_rows = timesteps - 1  # Number of rows to add as padding

            # Pad the matrix along the rows
            padded_matrix = np.pad(matrix_temporal, ((padding_rows, 0), (0, 0)), mode='constant', constant_values=padding_value)

            for j,z1,z2 in zip(range(len(matrix_temporal) - timesteps + 1),temp_col,Attack): # Ensures that extracted substrings have uniform length of timesteps and do not go outside the original sequence boundary. Avoid extracting incomplete substring
                data = {'Value': [padded_matrix[j:(j + timesteps)]],
                        'ProcessGuid': [i],
                        'UtcTime':z1,
                        "numVentana":j,
                        "Attack":z2}
                df = pd.DataFrame(data)
                Xs=pd.concat([Xs,df])
        return Xs
    else:
        Xs = []
        for i in X.ProcessGuid.unique():
            matrix_temporal = X[X.ProcessGuid == i].sort_values(by="UtcTime").drop(["ProcessGuid","UtcTime"], axis=1).values

            # Define padding values and amount
            padding_value = -1  # Change this to the value you want for padding
            padding_rows = timesteps - 1  # Number of rows to add as padding

            # Pad the matrix along the rows
            padded_matrix = np.pad(matrix_temporal, ((padding_rows, 0), (0, 0)), mode='constant', constant_values=padding_value)

            for j in range(len(matrix_temporal) - timesteps + 1): # Ensures that extracted substrings have uniform length of timesteps and do not go outside the original sequence boundary. Avoid extracting incomplete substring
                Xs.append(padded_matrix[j:(j + timesteps)])

        return np.array(Xs)



input_file = sys.argv[1]
X_train = pd.read_csv(input_file)

timesteps = timesteps_calculation(X_train, timesteps_max)
X_train = reshaping_data(X_train, timesteps=timesteps,test=False)

input_shape = (X_train.shape[1], X_train.shape[2])
initial_seq = Sequential()
initial_seq.add(Input(shape=input_shape))
initial_seq.add(Masking(mask_value=-1)) # Must match padding_value 
input_seq = initial_seq.input
x = initial_seq.output

output = TimeDistributed(Dense(input_shape[-1], activation="sigmoid"))(x)
lstm_autoencoder = Model(inputs=input_seq, outputs=output)

save_flattened_model_params("lstm_init.npy", lstm_autoencoder)
array = np.load('lstm_init.npy')
print(array.shape)

    # lstm_autoencoder_conf = model_conf["lstm-autoencoder"]

    # encoder_conf = lstm_autoencoder_conf["encoder"]

    # # Encoder
    # for i in range(encoder_conf["n-layers"]):
    #     layer = layer = encoder_conf[str(i)]
    #     x = add_layer(layer, x, X_train.shape if layer["type"] in ["repeat-vector", "time-distributed"] else None)

    # # Decoder
    # decoder_conf = lstm_autoencoder_conf["decoder"]

    # for j in range(decoder_conf["n-layers"]):
    #     layer = decoder_conf[str(j)]
    #     x = add_layer(layer, x, X_train.shape if layer["type"] in ["time-distributed"] else None)