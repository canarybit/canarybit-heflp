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
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from logging import WARNING
import numpy as np
from typing import Callable, List, Tuple, Union, Optional, Dict
from functools import reduce
from enum import Enum
import random

from heflp.secureproto.homoencrypschemes.flashev2 import Flashev2CypherParams, Flashev2CTxt
from heflp.secureproto.quantization.mwavg import MWAvgParams
from heflp.secureproto.quantization.encode import decode_all_layer_info, encode_alpha_list
from heflp.secureproto.quantization.alpha import decide_alpha_per_layer
from heflp.utils import logger
from heflp.utils.perf import TimeMarker
import heflp

if heflp.info.EVALUATION_MODE:
    LOGGER_EVAL = logger.getEvalLogger()
T_MARKER = TimeMarker()
LOGGER = logger.getLogger()

class Flashev2Property(Enum):
    MWAVG_PARAMS = "1"
    FLASHEV2_PARAMS = "2"
    M_PRIME = "3"
    LAYER_INFO = "4"
    ALPHA_LIST = "5" # For quantization

class Flashev2Strategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        model_len: int,
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
        incremental_mode: bool = True,
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
        self.model_len = model_len
        self.alpha_list = None # The range of model parameters for quantization [-alpha, alpha] per layer
        self.m_prime = 1. # The value to divide the degree before MWAvg
        self.mwavg_param_bytes = None
        self.shuffled_id_list = []  # The shuffled client ID list, used to create S for Flashev2
        self.incremental_mode = incremental_mode

    def _create_S(self, server_round:int, client_id:int):
        if client_id < len(self.shuffled_id_list)-1:
            S = {
                Flashev2CypherParams.concat_prefix(server_round, self.shuffled_id_list[client_id]): 1,
                Flashev2CypherParams.concat_prefix(server_round, self.shuffled_id_list[client_id+1]): -1,
            }
        else:
            S = {
                Flashev2CypherParams.concat_prefix(server_round, client_id): 1,
                Flashev2CypherParams.concat_prefix(server_round, client_id+1): -1,
            }
        return S

    def _configure_flashev2(
        self, server_round, client_id: int, config: Dict[str, Scalar]
    ):
        flashev2_params = Flashev2CypherParams(0, self.model_len, self._create_S(server_round, client_id))
        config[Flashev2Property.FLASHEV2_PARAMS.value] = flashev2_params.to_bytes()
        return config

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        if self.mwavg_param_bytes:
            config[Flashev2Property.MWAVG_PARAMS.value] = self.mwavg_param_bytes
        if self.alpha_list:
            config[Flashev2Property.ALPHA_LIST.value] = encode_alpha_list(self.alpha_list)
        config[Flashev2Property.M_PRIME.value] = self.m_prime

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        ret = []
        self.shuffled_id_list= list(range(len(clients)+1))
        random.shuffle(self.shuffled_id_list)
        for i, client in enumerate(clients):
            config = self._configure_flashev2(server_round, i, config.copy())
            ret.append((client, FitIns(parameters, config)))

        LOGGER.debug(f"Sampled clients:{[client.cid for client in clients]}")

        # Return client/config pairs
        return ret

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
        if self.mwavg_param_bytes:
            config[Flashev2Property.MWAVG_PARAMS.value] = self.mwavg_param_bytes
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

    def _aggregate(self, results:List[Tuple[Flashev2CTxt, int]]):
        """Compute weighted average."""
        weighted_results = [arr * miM for arr, miM in results]
        added_rst: Flashev2CTxt = reduce(lambda x,y: x+y, weighted_results)
        return added_rst

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

        param_list = [MWAvgParams.from_bytes(fit_res.metrics.get(Flashev2Property.MWAVG_PARAMS.value)) for _, fit_res in results]
        weights_results = [
            Flashev2CTxt.from_bytes(fit_res.parameters.tensors[0])
            for _, fit_res in results
        ]

        if self.incremental_mode:
            model_aggregated, comb_param = incremental_aggregate(weights_results, param_list)
            model_aggregated = model_aggregated.to_bytes()
        else:
            comb_param = MWAvgParams.combine(param_list)
            miM_list = comb_param.calculate_miM_list(param_list)
            model_aggregated = self._aggregate(zip(weights_results, miM_list)).to_bytes()

        max_d = max([param.degree  for param in param_list])
        max_m = max([param.m  for param in param_list])
        self.m_prime = (max_d + max_m)/2 * self.m_prime
        
        self.mwavg_param_bytes = comb_param.to_bytes()

        # Convert results
        parameters_aggregated = Parameters([model_aggregated], "Flashev2")

        T_MARKER.mark("Aggregation_done")
        if heflp.info.EVALUATION_MODE:
            LOGGER_EVAL.debug(f"Time overhead | {T_MARKER.get_all_intervals()}")

        v_ranges_list = []
        layer_sz_list = []
        for _, fit_res in results:
            arr = fit_res.metrics.get(Flashev2Property.LAYER_INFO.value)
            vr, lz = decode_all_layer_info(arr)
            v_ranges_list.append(vr)
            layer_sz_list.append(lz)

        def get_v_max_min(vranges_comb):
            max_values = np.max(np.max(vranges_comb, axis=0), axis=1)
            min_values = np.min(np.min(vranges_comb, axis=0), axis=1)
            v_max_min = list(zip(max_values, min_values))
            return v_max_min
        
        v_max_min = get_v_max_min(v_ranges_list)
        layer_sz_comb = np.sum(layer_sz_list, axis=0)
        LOGGER.debug(f"Combined Value Ranges: {v_max_min}")
        LOGGER.debug(f"Combined Layer Sizes: {layer_sz_comb}")
        self.alpha_list = decide_alpha_per_layer(v_max_min, layer_sz_comb)
        LOGGER.debug(f"New Alpha List: {self.alpha_list}")

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
def incremental_aggregate(results_lst:List[Flashev2CTxt], mwavg_lst:List[MWAvgParams]):
    """Aggregate Flashev2CTxt in an incremental way"""
    assert len(results_lst) == len(mwavg_lst)
    assert len(results_lst) >= 1, "The List of items is empty. At least one item is required for aggregation"
    M = mwavg_lst[0].m
    sum_d = mwavg_lst[0].degree
    sumup = results_lst[0]
    for i in range(len(results_lst)):
        if i == 0:
            continue
        if mwavg_lst[i].m < M:
            tmp = sumup*int(M//mwavg_lst[i].m)
            # tmp = avg*p_s[i].m/M
            sumup = tmp + results_lst[i]
            M = mwavg_lst[i].m
        else:
            sumup = sumup + results_lst[i] * int(mwavg_lst[i].m // M)
        sum_d += mwavg_lst[i].degree
    p_comb_old = MWAvgParams(sum_d, M)
    return sumup, p_comb_old