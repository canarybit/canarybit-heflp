from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Status, Code, Parameters
import flwr as fl
import numpy as np

from heflp.strategy.bfv import BFVProperty
from heflp.secureproto.homoencrypschemes.bfv import BFVCypher
from heflp.secureproto.quantization.mwavg import MWAvgQuantizer, MWAvgParams
from heflp.secureproto.quantization.quantizer import Quantizer
from heflp.training.params import flatten_model_params, unflatten_model_params
from heflp.training.runner import Runner
from heflp.utils import logger
from heflp.utils.perf import TimeMarker
import heflp

if heflp.info.EVALUATION_MODE:
    LOGGER_EVAL = logger.getEvalLogger()
T_MARKER = TimeMarker()
LOGGER = logger.getLogger()

class BFVClient(fl.client.Client):
    def __init__(self, cypher:BFVCypher,quantizer:MWAvgQuantizer, model, runner:Runner, fit_epochs:int=1) -> None:
        super().__init__()
        self.cypher = cypher
        self.model = model
        self.quantizer = quantizer
        self.runner = runner
        self.max_length = len(flatten_model_params(model))
        self.fit_epochs = fit_epochs

    def _extract_model(self, model_params:Parameters, mwavg_param_bytes: bytes):
        if mwavg_param_bytes:
            T_MARKER.mark("decryption_start")
            encrypted_ret = self.cypher.from_bytes(model_params.tensors)
            q_ret = self.cypher.decrypt(encrypted_ret, self.max_length)
            T_MARKER.mark("decryption_done")
            T_MARKER.mark("dequantization_start")
            dm = MWAvgParams.from_bytes(mwavg_param_bytes)
            ret = self.quantizer.dequantize(q_ret, dm)
            T_MARKER.mark("dequantization_done")
        else:
            LOGGER.info("Initialize model")
            ret = fl.common.parameters_to_ndarrays(model_params)[0]
        LOGGER.debug(f"Received global model param [-10:0]: {ret[-10:]}")
        unflatten_model_params(ret, self.model)

    def fit(self, ins: FitIns) -> FitRes:
        LOGGER.info("Fit start")
        if BFVProperty.HE_CONTEXT.value in ins.config:
            self.cypher.set_context_bytes(ins.config.get(BFVProperty.HE_CONTEXT.value))
        self._extract_model(ins.parameters, ins.config.get(BFVProperty.MWAVG_PARAMS.value, None))
        T_MARKER.reset() # Reset the time marker
        T_MARKER.mark("training_start")
        self.runner.train(self.model, epochs=self.fit_epochs)
        T_MARKER.mark("training_done")
        n_i = self.runner.get_dataset_size('train')
        flattened_ret = flatten_model_params(self.model)
        LOGGER.debug(f"Trained local model param [-10:0]: {flattened_ret[-10:]}")
        LOGGER.debug(f"Degree n: {n_i}")
        T_MARKER.mark("quantization_start")
        q_ret = self.quantizer.quantize(flattened_ret, n_i)
        param: MWAvgParams = self.quantizer.pop_cached_param()
        T_MARKER.mark("quantization_done")
        T_MARKER.mark("encryption_start")
        encrypted_tensor = self.cypher.encrypt(q_ret)
        encrypted_tensor_bytes = self.cypher.to_bytes(encrypted_tensor)
        T_MARKER.mark("encryption_done")
        LOGGER.info("Fit done")
        T_MARKER.mark("transmit")
        return FitRes(
            Status(Code.OK, 'BFV encrypted'),
            Parameters(encrypted_tensor_bytes, 'BFV'),
            n_i,
            {BFVProperty.MWAVG_PARAMS.value: param.to_bytes()}    
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        T_MARKER.mark("received")
        LOGGER.info("Evaluate start")
        if BFVProperty.HE_CONTEXT.value in ins.config:
            self.cypher.set_context_bytes(ins.config.get(BFVProperty.HE_CONTEXT.value))
        self._extract_model(ins.parameters, ins.config.get(BFVProperty.MWAVG_PARAMS.value, None))
        if heflp.info.EVALUATION_MODE:
            LOGGER_EVAL.debug(f"Time overhead | {T_MARKER.get_all_intervals()}") # Log the intervals for perf evaluation
        loss, accuracy = self.runner.test(self.model)
        LOGGER.info("Evaluate done")
        return EvaluateRes(
            Status(Code.OK, 'success'),
            loss,
            self.runner.get_dataset_size('test'),
            {"accuracy": accuracy}
        )