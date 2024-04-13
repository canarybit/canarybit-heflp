import flwr as fl
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
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
from typing import Callable, List, Tuple, Union, Optional, Dict
from functools import reduce
import numpy as np
from logging import WARNING

from heflp.secureproto.homoencrypschemes.flashe import FlasheCypher, FlasheCypherParams
from heflp.utils import logger
from heflp.utils.perf import TimeMarker
import heflp

if heflp.info.EVALUATION_MODE:
    LOGGER_EVAL = logger.getEvalLogger()
T_MARKER = TimeMarker()
LOGGER = logger.getLogger()

class FlasheStrategy(fl.server.strategy.FedAvg):
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
        self.global_flashe_params = FlasheCypherParams.create(end=model_len)
        self.sum_weights = 0

    def _get_flashe_params(self, client: ClientProxy):
        index_prefix_dir = {
            client.properties["index_prefix_1"]: 1,
            client.properties["index_prefix_2"]: -1,
        }
        return FlasheCypherParams.create(
            end=self.global_flashe_params.end, index_prefix_dir=index_prefix_dir
        )

    def _configure_flashe(
        self, server_round, seed: int, client: ClientProxy, config: Dict[str, Scalar]
    ):
        config["decrypt"] = self.global_flashe_params.serialize()
        client.properties["index_prefix_1"] = FlasheCypher.concat_prefix(
            server_round, seed
        )
        client.properties["index_prefix_2"] = FlasheCypher.concat_prefix(
            server_round, seed + 1
        )
        flashe_params = self._get_flashe_params(client)
        config["encrypt"] = flashe_params.serialize()
        return config

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        config['sum_weights'] = self.sum_weights

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        ret = []
        for i, client in enumerate(clients):
            config = self._configure_flashe(server_round, i, client, config.copy())
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
        config["decrypt"] = self.global_flashe_params.serialize()
        config['sum_weights'] = self.sum_weights
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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        T_MARKER.reset()
        T_MARKER.mark("Aggregation_start")
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        weights_results = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        ndarray_aggregated = reduce(np.add, weights_results)
        aggregated_parameters = ndarrays_to_parameters(ndarray_aggregated)

        self.sum_weights = sum([ fit_res.num_examples for _, fit_res in results])

        self.global_flashe_params.index_prefix_dir.clear()
        for client, _ in results:
            b = self._get_flashe_params(client)
            self.global_flashe_params.multi_add_update(
                b, 1
            )
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

        return aggregated_parameters, metrics_aggregated