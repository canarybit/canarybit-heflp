import warnings
import flwr as fl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from models import FCN
import argparse

# Initialize the logger here before importing the clients
# so that the strategy will also use the same logger saving logs to the same file.
from heflp.utils import logger
logfile = f"{logger.create_id_by_timestamp()}-client-mnist.log"
LOGGER =  logger.getLogger(logfile=f"./.tmp/logs/{logfile}")
logevalfile = f"{logger.create_id_by_timestamp()}-client-mnist-eval.log"
LOGGER_EVAL =  logger.getEvalLogger(logfile=f"./.tmp/logs/{logevalfile}")
from heflp.secureproto.homoencrypschemes.flashe import FlasheCypher
from heflp.secureproto.homoencrypschemes.flashev2 import Flashev2Cypher
from heflp.secureproto.homoencrypschemes.ckks import CKKSCypher
from heflp.secureproto.homoencrypschemes.bfv import BFVCypher
from heflp.secureproto.quantization.mwavg import MWAvgQuantizer, MWAvgLayerQuantizer
from heflp.training import params
from heflp.training.runner import PytorchRunner, FakeRunner, static_weight_generator
from heflp.client import FlasheClient, CKKSClient, BFVClient, BasicClient, Flashev2Client
from heflp import SUPPORT_SCHEMES, start_client

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Split the MNIST dataset into two parts (50% each)
def split_mnist_dataset(split_num=2):
    full_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
    step = len(full_dataset)//split_num
    dataset_splits = random_split(full_dataset, [step]*split_num)
    
    return dataset_splits

# Load data for one client
def load_data(cid, n_splits):
    """Load MNIST (training and test set)."""
    dataset_splits = split_mnist_dataset(n_splits)
    trainset = dataset_splits[int(cid)]
    testset = MNIST(root='./data', train=False, transform=ToTensor(), download=True)

    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default='basic', choices=SUPPORT_SCHEMES, help=f'Homomorphic encryption mode (default basic), support {SUPPORT_SCHEMES}')
    parser.add_argument("-c", "--cid", type=int, default=0, help='Client id, mandatory')
    parser.add_argument("-n", "--total_num_clients", default=2, type=int, help='Total number of clients (default 2)')
    parser.add_argument("-e", "--epochs_per_round", type=int, default=1, help='Epochs for each round, default 1')
    parser.add_argument("-a", "--address", type=str, default="127.0.0.1:8080", help='Server address address:port (default localhost:8080)')
    parser.add_argument("--ca", type=str, default=".tmp/certificates/ca.crt", help='CA certificate file')
    parser.add_argument("-C", "--comment", type=str, default="", help='Comment for this process, will be added to the meta data and log')
    args = parser.parse_args()
    mode = args.mode
    server_addr =  args.address
    epochs_per_round = args.epochs_per_round
    cid = args.cid
    total_num_clients = args.total_num_clients
    
    meta = {
        "mode": mode,
        "cid": cid,
        "total_num_clients": total_num_clients,
        "comment": args.comment,
    }
    LOGGER.info(f"Meta | {meta}")
    LOGGER_EVAL.info(f"Meta | {meta}")

    model = FCN().to(DEVICE)
    
    # Define the runner
    # Here I also wrote a FakeRunner to support debugging without real training.
    if_training = True
    if if_training:
        '''If standard training'''
        trainloader, testloader = load_data(cid, total_num_clients)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metric_fn = lambda outputs, labels: (torch.max(outputs.data, 1)[1] == labels).sum().item()
        runner = PytorchRunner(trainloader, testloader, DEVICE, criterion, optimizer, metric_fn)
    else:
        '''If test only:'''
        train_gen = static_weight_generator(1000)
        test_gen = static_weight_generator(1000)
        runner = FakeRunner(train_gen, test_gen)

    # Initialize the Heflp client
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
    
    start_client(server_addr, client, args.ca)

    if cid == 0:
        from heflp.training.params import save_flattened_model_params
        save_flattened_model_params("outputs/refined_fcn.npy", model)