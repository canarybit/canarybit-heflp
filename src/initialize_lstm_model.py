from models import FCN
import numpy as np
from heflp.training.params import save_flattened_model_params
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Masking, Dense, Dropout, LSTM, RepeatVector, TimeDistributed, Bidirectional

import sys
import pandas as pd

input_file = sys.argv[1]
df = pd.read_csv(input_file)
input_shape = (df.shape[1], df.shape[2])
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