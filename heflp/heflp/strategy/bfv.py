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

from heflp.secureproto.homoencrypschemes.bfv import BFVHelper
from heflp.secureproto.quantization.mwavg import MWAvgParams
from heflp.utils import logger
from heflp.utils.perf import TimeMarker
import heflp

if heflp.info.EVALUATION_MODE:
    LOGGER_EVAL = logger.getEvalLogger()
T_MARKER = TimeMarker()
LOGGER = logger.getLogger()

class BFVProperty(Enum):
    HE_CONTEXT = '1'
    MWAVG_PARAMS = "2"

class BFVStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        n=2**14,
        t=2**19+1,
        t_bits=20,
        sec=128,
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
        bfv_params = {
            'scheme': 'BFV',    # can also be 'bfv'
            'n': n,         # Polynomial modulus degree, the num. of slots per plaintext,
                                #  of elements to be encoded in a single ciphertext in a
                                #  2 by n/2 rectangular matrix (mind this shape for rotations!)
                                #  Typ. 2^D for D in [10, 16]
            't': t,         # Plaintext modulus. Encrypted operations happen modulo t
                                #  Must be prime such that t-1 be divisible by 2^N.
            't_bits': t_bits,       # Number of bits in t. Used to generate a suitable value
                                #  for t. Overrides t if specified.
            'sec': sec,         # Security parameter. The equivalent length of AES key in bits.
                                #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
                                #  More means more security but also slower computation.
        }
        self.he = BFVHelper(bfv_params)
        self.mwavg_param_bytes = None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        config[BFVProperty.HE_CONTEXT.value] = self.he.get_context_bytes()
        if self.mwavg_param_bytes:
            config[BFVProperty.MWAVG_PARAMS.value] = self.mwavg_param_bytes
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
        config[BFVProperty.HE_CONTEXT.value] = self.he.get_context_bytes()
        if self.mwavg_param_bytes:
            config[BFVProperty.MWAVG_PARAMS.value] = self.mwavg_param_bytes
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
        added_results = [arr * miM for arr, miM in results]
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

        param_list = [MWAvgParams.from_bytes(fit_res.metrics.get(BFVProperty.MWAVG_PARAMS.value)) for _, fit_res in results]
        comb_param = MWAvgParams.combine(param_list)
        self.mwavg_param_bytes = comb_param.to_bytes()
        miM_list = comb_param.calculate_miM_list(param_list)

        # Convert results
        weights_results = [
            np.array(self.he.from_bytes(fit_res.parameters.tensors))
            for _, fit_res in results
        ]
        ndarray_aggregated = self.he.to_bytes(self._aggregate(zip(weights_results, miM_list)).tolist())
        parameters_aggregated = Parameters(ndarray_aggregated, "BFV")

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