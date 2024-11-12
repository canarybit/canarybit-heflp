import pandas as pd

import plotly
import plotly.express as px
from pylab import *

from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import keras.optimizers
import tensorflow
tensorflow.random.set_seed(42)
from keras.layers import Input, Dense, Dropout, LSTM, RepeatVector, TimeDistributed, Bidirectional, Attention
from keras import regularizers
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences

import re
import os
import numpy as np
from numpy.typing import NDArray


# LSTM Autoencoder model configuration
DEFAULT_MODEL_CONF = {
    "lstm-autoencoder": {
        "encoder": 
            {
                "n-layers": 5,
                "0": {"type": "lstm", "size":32, "activation": "tanh", "return-sequences":True},
                "1": {"type": "dropout", "rate":0.2},
                "2": {"type": "lstm", "size":8, "activation": "tanh", "return-sequences":False},
                "3": {"type": "dropout", "rate":0.2},
                "4": {"type": "repeat-vector"}
            },
        "decoder": 
            {
                "n-layers": 5,
                "0": {"type": "lstm", "size":8, "activation": "tanh", "return-sequences":True},
                "1": {"type": "dropout", "rate":0.2},
                "2": {"type": "lstm", "size":32, "activation": "tanh", "return-sequences":True},
                "3": {"type": "dropout", "rate":0.2},
                "4": {"type": "time-distributed"}
            },
        "optimizer": "adam",
        "learning-rate": 0.001,
        "loss": "mse",
        "epochs": 50,
        "batch-size": 25
    }
}

def reading_files(path:str)->pd.DataFrame:
    
    """
    Read csv files, drop 'Unnamed: 0' column and set 'UtcTime' as index.
    
    Args:
        path (str): Preprocessed train-test input files.
        
    Returns:
        df (pandas.DataFrame): Preprocessed DataFrame.
    """
    
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Attack' in df.columns:
        df.Attack = df.Attack.astype(str)
    df=df.reset_index(drop=True)
    
    return df

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

def data_generator(input_data:NDArray, batch_size:int, timesteps:int, n_batches:int):
    
    """
    Generates batches of training data for LSTM Autoencoder model.
    
    Args:
        input_data (ndarray): Input data array of shape (samples, sequence_length, features).
        batch_size (int): Size of each batch.
        timesteps (int): Desired length of each sequence in a batch.
        
    Yields:
        tuple: A tuple containing the input batch and the corresponding target batch.
    
    """

    input_data = input_data[:n_batches*batch_size]

    while True:
        for batch in range(0, n_batches):

            current_batch = input_data[batch*batch_size:(batch+1)*batch_size]
            current_batch = pad_sequences(current_batch, maxlen=timesteps)
            print(current_batch.shape)

            yield current_batch, current_batch

def create_lstm_autoencoder(shape:Tuple[int,int], model_conf:dir):
    
    """
    Create the LSTM Autoencoder model according to the configured parameters.
    
    Args:
        shape (tuple of 2 int): Input training data array (normal data, no attacks) shape (sequence_length, features).
        model_configuration (dict): Configurable model parameters.
        
    Returns:
        lstm_autoencoder (keras.engine.training.Model): LSTM Autoencoder trained model.
    
    """

    lstm_autoencoder = Sequential()

    # Encoder
    n_layers = model_conf["lstm-autoencoder"]["encoder"]["n-layers"]

    for i in range(n_layers):
        layer = model_conf["lstm-autoencoder"]["encoder"][str(i)]

        if layer["type"] == "lstm":
            if i==0:
                lstm_autoencoder.add(LSTM(units=layer["size"], 
                                          activation=layer["activation"], 
                                          return_sequences=layer["return-sequences"], 
                                          activity_regularizer=regularizers.l2(model_conf["lstm-autoencoder"]["learning-rate"]),
                                          input_shape=(shape[0],shape[1]), kernel_regularizer=regularizers.l2(model_conf["lstm-autoencoder"]["learning-rate"])))
            else:
                lstm_autoencoder.add(LSTM(units=layer["size"], activation=layer["activation"], return_sequences=layer["return-sequences"]))
        
        elif layer["type"] == "bidirectional-lstm":
            if i==0:
                lstm_autoencoder.add(Bidirectional(LSTM(units=layer["size"],
                                                        activation=layer["activation"],
                                                        return_sequences=layer["return-sequences"], 
                                                        activity_regularizer=regularizers.l2(model_conf["lstm-autoencoder"]["learning-rate"]),
                                                        input_shape=(shape[0],shape[1]), 
                                                        kernel_regularizer=regularizers.l2(model_conf["lstm-autoencoder"]["learning-rate"]))))
            else:
                lstm_autoencoder.add(Bidirectional(LSTM(units=layer["size"], 
                                                        activation=layer["activation"], 
                                                        return_sequences=layer["return-sequences"])))

        elif layer["type"] == "dropout":
             lstm_autoencoder.add(Dropout(rate=layer["rate"]))

        elif layer["type"] == "repeat-vector":
            lstm_autoencoder.add(RepeatVector(shape[0]))
            
        elif layer["type"] == "attention":
            lstm_autoencoder.add(Attention())

    # Decoder
    n_layers = model_conf["lstm-autoencoder"]["decoder"]["n-layers"]

    for j in range(n_layers):
        layer = model_conf["lstm-autoencoder"]["decoder"][str(j)]

        if layer["type"] == "lstm":
            lstm_autoencoder.add(LSTM(units=layer["size"], activation=layer["activation"], return_sequences=layer["return-sequences"]))

        elif layer["type"] == "dropout":
             lstm_autoencoder.add(Dropout(rate=layer["rate"]))

        elif layer["type"] == "time-distributed":
            lstm_autoencoder.add(TimeDistributed(Dense(shape[1])))

    return lstm_autoencoder

def lstm_autoencoder_prediction_and_errors(lstm_autoencoder, X_train, X_test, columns):
    
    """
    Predicts the input values of the training and test data, storing the error obtained for each of them.
    
    Args:
        lstm_autoencoder (keras.engine.training.Model): Length of each sequence in a batch.
        X_train (ndarray): Input training data array of shape (samples, sequence_length, features).
        X_test (ndarray): Input test data array of shape (samples, sequence_length, features).
        columns (list): List of X_test column names. 
        
    Returns:
        mean_obs_mse_train (pandas.DataFrame): Mean value of the obtained errors in each variable of the training data.
        mean_obs_mse_test (pandas.DataFrame): Mean value of the obtained errors in each variable of the test data.
        mse_test_df (pandas.DataFrame): DataFrame with the errors obtained for all variables and observations (rows).
        pred_test (ndarray): Predicted values for test data.
    """

    pred_train = lstm_autoencoder.predict(X_train)
    pred_test = lstm_autoencoder_test_prediction(lstm_autoencoder=lstm_autoencoder, X_test=X_test)

    mse_train = np.mean(np.power(X_train - pred_train, 2), axis=1)
    mse_train_df = pd.DataFrame(mse_train, columns=columns)

    mse_test = np.mean(np.power(X_test - pred_test, 2), axis=1)
    mse_test_df = pd.DataFrame(mse_test, columns=columns)

    mean_obs_mse_train_list = [mse_train_df.iloc[i].mean() for i in range(len(mse_train_df))]
    mean_obs_mse_test_list = [mse_test_df.iloc[i].mean() for i in range(len(mse_test_df))]

    mean_obs_mse_train = pd.DataFrame(mean_obs_mse_train_list, columns=["mean_mse"])

    mean_obs_mse_test = pd.DataFrame(mean_obs_mse_test_list, columns=["mean_mse"])

    return mean_obs_mse_train, mean_obs_mse_test, mse_test_df, pred_test

def lstm_autoencoder_test_prediction(lstm_autoencoder, X_test):
    
    """
    Predicts the input values of the training and test data, storing the error obtained for each of them.
    
    Args:
        lstm_autoencoder (keras.engine.training.Model): Length of each sequence in a batch.
        X_test (DataFrame): Input test data frame.   
    Returns:
        X_test (DataFrame): Input test data frame.
    """

    X_test_values = np.array(X_test["Value"].tolist())
    X_test_values = X_test_values.astype("float32")
    predictions = lstm_autoencoder.predict(X_test_values)
    
    return predictions

def anomalies_explanation(
        mean_obs_mse_test, 
        mse_test_df, 
        anom_obs_percentiles_conf, 
        anom_variables_percentiles_conf):
    
    """
    When an anomaly is detected, this function explains which variable or variables are responsible. For this purpose,
    the variable or variables with a significantly high error are selected.
    
    Args:
        mean_obs_mse_test (pandas.DataFrame): Mean value of the obtained errors in each variable of the test data.
        mse_test_df (pandas.DataFrame): DataFrame with the errors obtained for all variables and observations (rows).
        
    Returns:
        mse_df (pandas.DataFrame): DataFrame with the errors obtained for all variables and observations, the "mean_mse" column
                                   with the mean value of errors for each row, the "anom" binary column (0: no anomalous, 
                                   1: anomalous) and the "anom_columns_list" with the list of variables explaining the anomaly. 
    """

    mean_obs_mse_test["anom"] = 0
    
    # IQR METHOD to detect anomalous rows
    q1, q3 = np.percentile(mean_obs_mse_test.mean_mse, anom_obs_percentiles_conf)
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    outlier_obs = mean_obs_mse_test[mean_obs_mse_test.mean_mse > upper_bound]
    upp_perc = anom_obs_percentiles_conf[1]
    while len(outlier_obs) == 0:
        upp_perc -= 1
        new_iqr_anomalies = [anom_obs_percentiles_conf[0], upp_perc]
        q1, q3 = np.percentile(mean_obs_mse_test.mean_mse, new_iqr_anomalies)
        iqr = q3 - q1
        upper_bound = q3 + (1.5 * iqr)
        outlier_obs = mean_obs_mse_test[mean_obs_mse_test.mean_mse > upper_bound]
        if (len(outlier_obs) > 10) or (upp_perc < 25):
            q1, q3 = np.percentile(mean_obs_mse_test.mean_mse, anom_obs_percentiles_conf)
            iqr = q3 - q1
            upper_bound = q3 + (1.5 * iqr)
            outlier_obs = mean_obs_mse_test[mean_obs_mse_test.mean_mse > upper_bound]
            break
    print("\nNumber of outliers found:", len(outlier_obs), "\n")

    mean_obs_mse_test.loc[(mean_obs_mse_test.mean_mse > upper_bound), ["anom"]] = 1
    mse_df = mse_test_df.copy()
    mse_df["mean_mse"] = mean_obs_mse_test.mean_mse
    mse_df["anom"] = mean_obs_mse_test.anom
    mse_df["anom_columns_list"] = "None"
    mse_df2 = mse_df.drop(["mean_mse", "anom", "anom_columns_list"], axis=1)
    mse_df2.rename({"n-steps": "n-steps_"}, axis=1, inplace=True)
    mse_df2.rename({"CommandLine-len": "CommandLine-len_"}, axis=1, inplace=True)

    anom_test_index = list(mean_obs_mse_test[mean_obs_mse_test.anom == 1].index)

    for anom_time in anom_test_index:
        if isinstance(mse_df2.loc[anom_time], pd.Series):
            mse_df_i = mse_df2.loc[anom_time].to_frame().T
        else:
            mse_df_i = mse_df2.loc[anom_time]

        columns = mse_df_i.columns.to_list()
        if len(mse_df_i.index) > 1:
            mse_df_i = mse_df_i.groupby("UtcTime", axis=0).agg("mean")
        
        # IQR METHOD to detect anomalous variables
        if len(columns) > 2:
            q1, q3 = np.percentile(mse_df_i.loc[anom_time].values, anom_variables_percentiles_conf)
            iqr = q3 - q1
            upper_bound = q3 + (1.5 * iqr)
            outlier_cols_list = mse_df_i.columns[mse_df_i.loc[anom_time].gt(upper_bound)].tolist()
        else:
            max_val = mse_df_i.max().max()
            outlier_cols_list = list(mse_df_i.columns[mse_df_i.max() == max_val])

        if len(outlier_cols_list) > 0:
            final_outliers_list = []
            for var in outlier_cols_list:
                match = re.search("(?P<valor>.+?)_(.+|$)", str(var))
                final_outliers_list += [str(match[1])]
            outlier_cols_list = list(set(final_outliers_list))
        else:
            outlier_cols_list = ["None"]
        mse_df.loc[anom_time, ["anom_columns_list"]] = str(outlier_cols_list)

    mse_df.anom = mse_df.anom.apply(str)

    return mse_df


def anomaly_detected(mean_obs_mse_test, mse_test_df, df_test, X_test):
    
    """
    Detects anomalies and plots them  by the error obtained from each observation over time.
    
    Args:
        mean_obs_mse_test (pandas.DataFrame): Mean value of the obtained errors in each variable of the test data.
        mse_test_df (pandas.DataFrame): DataFrame with the errors obtained for all variables and observations (rows).
        
    Returns:
        df_test_final (pandas.DataFrame): Original test dataframe with an extra column "anomaly_degree", that contains the 
                                          normalized anomaly degree (0 - 1) for each observation.
    """
    
    anom_processes_list = []
    print("*******Anomaly detected in this process******* \n")

    mse_df = anomalies_explanation(mean_obs_mse_test, mse_test_df)

    color_discrete_map = {'0': 'green', '1': 'red'}
    max_mse_value = mse_df.mean_mse.max() + 1
    min_mse_value = -0.5
    fig = px.scatter(mse_df, x=mse_df.index, y="mean_mse", hover_name="anom_columns_list",
                     color="anom", color_discrete_map=color_discrete_map)
    fig.update_layout(
        title={
            "text": "MSE evolution in time ",
            "y": 0.95, "x": 0.5,
            "xanchor": "center", "yanchor": "top"},
        showlegend=True)
    fig.update_layout(yaxis_range=[min_mse_value, max_mse_value])
    
    fig.show()
    
    df_test_final = df_test[:len(X_test)].copy()
    df_test_final.reset_index(inplace=True)

#     remove_idx = []
#     for i in range(len(df_test_final)):
#         row = df_test_final.iloc[i,1:].values.tolist()
#         if len(set(row))==len(set([0,"0",0.0])):
#             remove_idx.append(i)

#     df_test_final.drop(remove_idx, inplace=True)
#     df_test_final.reset_index(inplace=True, drop=True)
    
    scaler = MinMaxScaler()
    mse_df_scal = scaler.fit_transform(mse_df[["mean_mse"]])
    mse_df_scal = pd.DataFrame(mse_df_scal, columns=["anomaly_degree"])
    
    df_test_final["anomaly_degree"] = mse_df_scal["anomaly_degree"].values
    mse_df = mse_df.sort_values(by="mean_mse", ascending=False)
    
    return df_test_final

# Split the training dataset into several pieces, each of which containing the data for one client
def split_training_dataset(df:pd.DataFrame, n:int=2):
    def split(sequence:list, sep):
        chunk = [sep]
        sequence.pop(0)
        for val in sequence:
            if val == sep:
                yield chunk
                chunk = [sep]
            else:
                chunk.append(val)
        yield chunk

    df_list = df.values.tolist()
    dflist_splits = list(split(df_list, df_list[0])) # the first row as delimiter
    len_splits = [len(i) for i in dflist_splits]
    # Split the df according to the process
    df_splits = []
    start = 0
    for size in len_splits:
        ss = df[start:start+size]
        df_splits.append(ss)
        start = start + size

    # Combine processes into preset n pieses
    idx_splits = np.array_split(np.arange(len(df_splits)), n)
    n_combined_splits = [pd.concat([df_splits[idx] for idx in idx_list]) for idx_list in idx_splits]
    # for s in n_combined_splits:
    #     print(s.index.tolist())
    #     print(len(s))

    return n_combined_splits
