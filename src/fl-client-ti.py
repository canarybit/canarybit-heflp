import warnings
import argparse
import math
# import keras  # uncomment this line might cause the program to be unexpectedly stuck, weird
from typing import List

# Initialize the logger here before importing the clients
# so that the strategy will also use the same logger saving logs to the same file.
from heflp.utils import logger
logfile = f"{logger.create_id_by_timestamp()}-client-it.log"
LOGGER =  logger.getLogger(logfile=f"./.tmp/logs/{logfile}")
logevalfile = f"{logger.create_id_by_timestamp()}-client-it-eval.log"
LOGGER_EVAL =  logger.getEvalLogger(logfile=f"./.tmp/logs/{logevalfile}")
from heflp.secureproto.homoencrypschemes.flashe import FlasheCypher
from heflp.secureproto.homoencrypschemes.flashev2 import Flashev2Cypher
from heflp.secureproto.homoencrypschemes.ckks import CKKSCypher
from heflp.secureproto.homoencrypschemes.bfv import BFVCypher
from heflp.secureproto.quantization.quantizer import Quantizer
from heflp.secureproto.quantization.mwavg import MWAvgQuantizer, MWAvgLayerQuantizer
from heflp.client import FlasheClient, CKKSClient, BFVClient, BasicClient, Flashev2Client

DIR = "/home/felix/thesis-2023-zekun/data/threat-intelligence-exchange/Data/1.DataSets/"
DATASET_LIST = ["cmd.exe", "powershell.exe", "winword.exe"]
FILE_LIST = ["test.csv", "training.csv", "X_test.csv", "X_train.csv"]

from threatintellidataset import *
from heflp.training.runner import TensorflowRunner, FakeRunner, static_weight_generator
from heflp.training import params
from heflp import SUPPORT_SCHEMES, start_client

warnings.filterwarnings("ignore", category=UserWarning)

# Load the training and evaluation data
def load_data(cid:int=0, n_splits:int=1, timesteps:int=10):
    X_train_path = os.path.join(DIR, DATASET_LIST[0], FILE_LIST[3])
    X_test_path = os.path.join(DIR, DATASET_LIST[0], FILE_LIST[2])
    original_test_path = os.path.join(DIR, DATASET_LIST[0], FILE_LIST[0])

    X_train = reading_files(X_train_path)
    X_test = reading_files(X_test_path)
    df_test = reading_files(original_test_path)

    cols = list(X_test.columns)
    if n_splits!=1:
        try:
            splits = split_training_dataset(X_train, n_splits)
            X_train = splits[cid]
        except:
            print("Split data failed!")
            exit()

    X_train = reshaping_data(X_train, timesteps=timesteps)
    X_test = reshaping_data(X_test, timesteps=timesteps)

    return X_train, X_test, df_test, cols

# Customize the Runner for training and evaluating the LSTM model
class LSTMRunner(TensorflowRunner):
    def __init__(
        self,
        X_train: NDArray,
        X_test: NDArray,
        df_test: pd.DataFrame,
        columns: List[str],
        criterion: str,
        optimizer: keras.optimizers.Optimizer,
        batch_size: int,
        timesteps: int = 0,
        timesteps_max: int = 10
    ) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.df_test = df_test
        self.columns = columns
        if timesteps > 0:
            self.timesteps = timesteps
        else:
            timesteps_calc = math.floor(len(X_train)/3)
            if timesteps_calc >= timesteps_max:
                self.timesteps = timesteps_max
            else:
                self.timesteps = timesteps_calc
        n_batches = len(self.X_train) // batch_size
        train_gen = data_generator(batch_size=batch_size, timesteps=self.timesteps, input_data=self.X_train, n_batches=n_batches)
        super().__init__(train_gen, None, n_batches, len(self.X_test), criterion, optimizer, None)

    def _test_full(self, model):
        """Validate the model on the test set."""
        (
            mean_obs_mse_train,
            mean_obs_mse_test,
            mse_test_df,
            pred_test,
        ) = lstm_autoencoder_prediction_and_errors(
            lstm_autoencoder=model,
            X_train=self.X_train,
            X_test=self.X_test,
            df_test=self.df_test,
            columns=list(self.columns),
        )
        return mean_obs_mse_train, mean_obs_mse_test, mse_test_df, pred_test

    def test(self, model):
        _, s1, _, _ = self._test_full(model)
        return s1.mean_mse.mean(), s1.mean_mse.mean()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default='basic', choices=SUPPORT_SCHEMES, help=f'Homomorphic encryption mode (default basic), support {SUPPORT_SCHEMES}')
    parser.add_argument("-c", "--cid", default=0, type=int, help='Client id, default=0')
    parser.add_argument("-n", "--total_num_clients", default=2, type=int, help='Total number of clients, default 2')
    parser.add_argument("-e", "--epochs_per_round", type=int, default=5, help='Epochs for each round, default 5')
    parser.add_argument("-a", "--address", type=str, default="127.0.0.1:8080", help='Server address address:port')
    parser.add_argument("--ca", type=str, default=".tmp/certificates/ca.crt", help='CA certificate file')
    parser.add_argument("-C", "--comment", type=str, default="", help='Comment for this process, will be added to the meta data and log')
    args = parser.parse_args()

    # Extract the parameters
    mode = args.mode
    cid = args.cid
    total_n = args.total_num_clients
    server_addr = args.address
    epochs_per_round = args.epochs_per_round

    # Log the parameters
    meta = {
        "mode": mode,
        "cid": args.cid,
        "total_num_clients": args.total_num_clients,
        "epochs_per_round": {epochs_per_round},
        "comment": args.comment,
    }
    LOGGER.info(f"Meta | {meta}")

    # Load the training data
    X_train, X_test, df_test, cols = load_data(cid, total_n)
    data_shape = (X_train.shape[1], X_train.shape[2])

    model = create_lstm_autoencoder(data_shape, DEFAULT_MODEL_CONF)

    # Define the runner
    # Here I also wrote a FakeRunner to support debugging without real training.
    if_training = True
    if if_training:
        '''If standard training'''
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        runner = LSTMRunner(X_train, X_test, df_test, cols, 'mse', optimizer, 100)
    else:
        '''If test only:'''
        train_gen = static_weight_generator(1000)
        test_gen = static_weight_generator(1000)
        runner = FakeRunner(train_gen, test_gen)

    # Initialize the Heflp client with the runner defined before.
    if mode == 'basic':
        client=BasicClient(model, runner, epochs_per_round)
    elif mode == 'flashe':
        seed = b'\xecv\xe3\x8b\x9b\xb3\x95j\xdb\x8a\xaa\x8a\nm4\xb7~wf\na!e]\x84E\x98s&P\xb2P'
        cypher = FlasheCypher(seed, bit_width=24)
        quantizer = MWAvgQuantizer(r_max=1, bit_width=16)
        client=FlasheClient(cypher, quantizer, model, runner, epochs_per_round)
    elif mode == 'ckks':
        cypher = CKKSCypher('./.tmp/ckks_priv.key', './.tmp/ckks_pub.key')
        client=CKKSClient(cypher, model, runner, epochs_per_round)
    elif mode == 'bfv':
        cypher = BFVCypher('./.tmp/bfv_priv.key', './.tmp/bfv_pub.key')
        quantizer = MWAvgQuantizer(r_max=1, bit_width=16)
        client=BFVClient(cypher, quantizer, model, runner, epochs_per_round)
    elif mode == 'flashev2':
        seed = b'\xecv\xe3\x8b\x9b\xb3\x95j\xdb\x8a\xaa\x8a\nm4\xb7~wf\na!e]\x84E\x98s&P\xb2P'
        cypher = Flashev2Cypher(seed, bit_width=16)
        layer_sizes = params.get_layer_sizes(params.flatten_model_params_per_layer(model))
        quantizer = MWAvgLayerQuantizer.create(layer_sizes, bit_width=16)
        client=Flashev2Client(cypher, quantizer, model, runner, epochs_per_round)
    else:
        raise ValueError(f"Not support HE mode {mode}! Please use flag -h for details")

    # Start the Heflp client
    start_client(server_addr, client, args.ca)

    if cid == 0:
        from heflp.training.params import save_flattened_model_params
        save_flattened_model_params("outputs/refined_lstm.npy", model)