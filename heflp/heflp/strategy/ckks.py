import flwr as fl
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    NDArray,
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from typing import Callable, List, Tuple, Union, Optional, Dict
import numpy as np
from functools import reduce
from logging import WARNING
from enum import Enum

from heflp.secureproto.homoencrypschemes.ckks import CKKSHelper
from heflp.utils import logger
from heflp.utils.perf import TimeMarker
import heflp

if heflp.info.EVALUATION_MODE:
    LOGGER_EVAL = logger.getEvalLogger()
T_MARKER = TimeMarker()
LOGGER = logger.getLogger()

class CKKSProperty(Enum):
    HE_CONTEXT = "1"
    INITIAL = "2"


class CKKSStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        n: int = 2**14,
        scale: int = 2**30,
        qi_sizes: List[int] = [60, 30, 30, 30, 60],
        fraction_fit: float = 1,
        fraction_evaluate: float = 1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        ckks_params = {
            'scheme': 'CKKS',   # can also be 'ckks'
            'n': n,         # Polynomial modulus degree. For CKKS, n/2 values can be
                                #  encoded in a single ciphertext.
                                #  Typ. 2^D for D in [10, 15]
            'scale': scale,     # All the encodings will use it for float->fixed point
                                #  conversion: x_fix = round(x_float * scale)
                                #  You can use this as default scale or use a different
                                #  scale on each operation (set in HE.encryptFrac)
            'qi_sizes': qi_sizes # Number of bits of each prime in the chain.
                                # Intermediate values should be  close to log2(scale)
                                # for each operation, to have small rounding errors.
        }
        self.he = CKKSHelper(ckks_params)
        self.initial = True

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        config[CKKSProperty.HE_CONTEXT.value] = self.he.get_context_bytes()
        config[CKKSProperty.INITIAL.value] = self.initial
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        LOGGER.debug(f"Sampled clients:{[client.cid for client in clients]}")

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        config[CKKSProperty.HE_CONTEXT.value] = self.he.get_context_bytes()
        config[CKKSProperty.INITIAL.value] = self.initial
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        LOGGER.debug(f"Sampled clients:{[client.cid for client in clients]}")

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def _aggregate(self, results:List[Tuple[NDArrays, int]]):
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])
        added_results = [arr * (num_examples / num_examples_total) for arr, num_examples in results]
        weights_prime: NDArray = reduce(np.add, added_results)
        return weights_prime

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        T_MARKER.reset()
        T_MARKER.mark("Aggregation_start")

        # Convert results
        weights_results = [
            (np.array(self.he.from_bytes(fit_res.parameters.tensors)), fit_res.num_examples)
            for _, fit_res in results
        ]
        model_aggregated = self.he.to_bytes(self._aggregate(weights_results).tolist())
        parameters_aggregated = Parameters(model_aggregated, "CKKS")

        if self.initial:
            self.initial = False

        T_MARKER.mark("Aggregation_done")
        if heflp.info.EVALUATION_MODE:
            LOGGER_EVAL.debug(f"Time overhead | {T_MARKER.get_all_intervals()}")

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated