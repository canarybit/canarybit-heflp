from typing import Callable, List, Tuple, Union, Optional, Dict
import flwr as fl
from flwr.common import (
    Metrics,
)
import argparse
import numpy as np

# Initialize the logger here before importing the strategies
# so that the strategy will also use the same logger saving logs to the same file.
from heflp.utils import logger
logfile = f"{logger.create_id_by_timestamp()}-server.log"
LOGGER =  logger.getLogger(logfile=f"./.tmp/logs/{logfile}")
logevalfile = f"{logger.create_id_by_timestamp()}-server-eval.log"
LOGGER_EVAL =  logger.getEvalLogger(logfile=f"./.tmp/logs/{logevalfile}")

from heflp.strategy import FlasheStrategy, CKKSStrategy, BFVStrategy, BasicStrategy, Flashev2Strategy
from heflp import SUPPORT_SCHEMES, start_server

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    LOGGER.info(f"accuracy: {sum(accuracies) / sum(examples)}")
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default='basic', choices=SUPPORT_SCHEMES, help=f'Homomorphic encryption mode (default basic), support {SUPPORT_SCHEMES}')
    parser.add_argument("-f", "--checkpoint_file", type=str, help='Model checkpoint file, .npy numpy file')
    parser.add_argument("-r", "--rounds", default=5, type=int, help='Number of Federated Learning rounds, default 5 rounds')
    parser.add_argument("-n", "--min_available_clients", default=2, type=int, help='Total number of clients, default 2')
    parser.add_argument("-C", "--comment", type=str, default="", help='Comment for this process, will be added to the meta data and log')
    parser.add_argument("--ca", type=str, default="./.tmp/certificates/ca.crt", help='CA certificate file')
    parser.add_argument("--ssl_priv_key", type=str, default="./.tmp/certificates/server.key", help='Server private key file')
    parser.add_argument("--ssl_pub_key", type=str, default="./.tmp/certificates/server.pem", help='Server public key file')
    args = parser.parse_args()
    mode = args.mode
    n_rounds = args.rounds
    min_available_clients = args.min_available_clients
    
    # Log the meta information
    meta = {
        "mode": mode,
        "rounds": n_rounds,
        "min_available_clients": min_available_clients,
        "comment": args.comment,
    }
    LOGGER.info(f"Meta | {meta}")
    LOGGER_EVAL.info(f"Meta | {meta}")

    try:
        init_param = np.load(args.checkpoint_file)
        length = len(init_param)
        initial_params = fl.common.ndarrays_to_parameters([init_param])
    except:
        raise ValueError(f"The checkpoint file does not exist or has a wrong format: {args.checkpoint_file}")

    init_param =  None if args.checkpoint_file == None else np.load(args.checkpoint_file)
    param_dict = {
        "min_available_clients": min_available_clients,
        "fraction_fit": 0.5,
        "fraction_evaluate": 0.5,
        "min_evaluate_clients": 1,
    }
    # Define the strategy
    if mode == 'basic':
        strategy = BasicStrategy(evaluate_metrics_aggregation_fn=weighted_average, initial_parameters=initial_params, **param_dict)
    elif mode == 'ckks':
        strategy = CKKSStrategy(evaluate_metrics_aggregation_fn=weighted_average, initial_parameters=initial_params, **param_dict)
    elif mode == 'bfv':
        strategy = BFVStrategy(evaluate_metrics_aggregation_fn=weighted_average, initial_parameters=initial_params, **param_dict)
    elif mode == 'flashe':
        strategy = FlasheStrategy(model_len=length, evaluate_metrics_aggregation_fn=weighted_average, initial_parameters=initial_params, **param_dict)
    elif mode == 'flashev2':
        strategy = Flashev2Strategy(model_len=length, evaluate_metrics_aggregation_fn=weighted_average, initial_parameters=initial_params, **param_dict)
    else:
        raise ValueError(f"Not support HE mode {mode}! Please use flag -h for details")

    # Start the Heflp server using the strategy defined above
    start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        root_certificate=args.ca,    # Enable SSL secure communication
        priv_key_path=args.ssl_priv_key,
        pub_key_path=args.ssl_pub_key,
    )
