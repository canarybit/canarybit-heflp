import warnings
import flwr as fl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToTensor, Compose, Normalize
from models import Net
import argparse

from heflp.utils import logger
logfile = f"{logger.create_id_by_timestamp()}-client-cifar.log"
LOGGER =  logger.getLogger(logfile=f"./.tmp/logs/{logfile}")
logevalfile = f"{logger.create_id_by_timestamp()}-client-cifar-eval.log"
LOGGER_EVAL =  logger.getEvalLogger(logfile=f"./.tmp/logs/{logevalfile}")
from heflp.secureproto.homoencrypschemes.flashe import FlasheCypher
from heflp.secureproto.homoencrypschemes.ckks import CKKSCypher
from heflp.secureproto.homoencrypschemes.bfv import BFVCypher
from heflp.secureproto.quantization.mwavg import MWAvgQuantizer, MWAvgLayerQuantizer
from heflp.training.runner import PytorchRunner, FakeRunner, static_weight_generator
from heflp.client import FlasheClient, CKKSClient, BFVClient, BasicClient, Flashev2Client
from heflp import SUPPORT_SCHEMES, start_client
from heflp.training import params

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)

    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

def create_init_model(model_type):
    if model_type == 'cnn':
        return Net()
    elif model_type == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_classes = 10  # CIFAR-10 has 10 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_type == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_classes = 10  # CIFAR-10 has 10 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_type == 'vgg11':
        model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        num_classes = 10  # CIFAR-10 has 10 classes
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    else:
        raise ValueError("Invalid model type")

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default='basic', choices=SUPPORT_SCHEMES, help=f'Homomorphic encryption mode (default basic), support {SUPPORT_SCHEMES}')
    parser.add_argument("-c", "--cid", type=int, default=0, help='Client id, mandatory')
    parser.add_argument("-M", "--model", type=str, default='cnn', help='Model type for training: cnn, resnet18, resnet50, vgg11')
    parser.add_argument("-n", "--total_num_clients", default=2, type=int, help='Total number of clients (default 2)')
    parser.add_argument("-a", "--address", type=str, default="127.0.0.1:8080", help='Server address address:port (default localhost:8080)')
    parser.add_argument("--ca", type=str, default=".tmp/certificates/ca.crt", help='CA certificate file')
    parser.add_argument("-C", "--comment", type=str, default="", help='Comment for this process, will be added to the meta data and log')
    args = parser.parse_args()
    mode = args.mode
    server_addr =  args.address
    cid = args.cid
    meta = {
        "mode": mode,
        "cid": cid,
        "total_num_clients": args.total_num_clients,
        "comment": args.comment,
    }
    LOGGER.info(f"Meta | {meta}")
    LOGGER_EVAL.info(f"Meta | {meta}")

    model = create_init_model(args.model)

    if_training = True
    if if_training:
        '''If standard training'''
        trainloader, testloader = load_data()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metric_fn = lambda outputs, labels: (torch.max(outputs.data, 1)[1] == labels).sum().item()
        runner = PytorchRunner(trainloader, testloader, DEVICE, criterion, optimizer, metric_fn)
    else:
        '''If test only:'''
        train_gen = static_weight_generator(1000)
        test_gen = static_weight_generator(1000)
        runner = FakeRunner(train_gen, test_gen)

    if mode == 'basic':
        fl.client.start_numpy_client(
            server_address=server_addr,
            client=BasicClient(model, runner),
        )
    elif mode == 'flashe':
        seed = b'\xecv\xe3\x8b\x9b\xb3\x95j\xdb\x8a\xaa\x8a\nm4\xb7~wf\na!e]\x84E\x98s&P\xb2P'
        cypher = FlasheCypher(seed, bit_width=16)
        quantizer = MWAvgQuantizer(r_max=1, bit_width=16)
        client=FlasheClient(cypher, quantizer, model, runner)
    elif mode == 'ckks':
        cypher = CKKSCypher('./.tmp/ckks_priv.key', './.tmp/ckks_pub.key')
        client=CKKSClient(cypher, model, runner)
    elif mode == 'bfv':
        cypher = BFVCypher('./.tmp/bfv_priv.key', './.tmp/bfv_pub.key')
        quantizer = MWAvgQuantizer(r_max=1, bit_width=16)
        client=BFVClient(cypher, quantizer, model, runner)
    elif mode == 'flashev2':
        seed = b'\xecv\xe3\x8b\x9b\xb3\x95j\xdb\x8a\xaa\x8a\nm4\xb7~wf\na!e]\x84E\x98s&P\xb2P'
        cypher = FlasheCypher(seed, bit_width=16)
        layer_sizes = params.get_layer_sizes(params.flatten_model_params_per_layer(model))
        quantizer = MWAvgLayerQuantizer.create(layer_sizes, bit_width=16)
        client=Flashev2Client(cypher, quantizer, model, runner)
    else:
        raise ValueError(f"Not support HE mode {mode}! Please use flag -h for details")

    start_client(server_addr, client, args.ca)

    if cid == 0:
        from heflp.training.params import save_flattened_model_params
        save_flattened_model_params(f"outputs/refined_{args.model}.npy", model)