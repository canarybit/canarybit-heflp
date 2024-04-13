import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Status, Code, Parameters

from heflp.strategy.flashev2 import Flashev2Property
from heflp.secureproto.homoencrypschemes.flashev2 import Flashev2Cypher, Flashev2CypherParams, Flashev2CTxt
from heflp.secureproto.quantization.quantizer import LayerQuantizer
from heflp.secureproto.quantization.mwavg import MWAvgParams, MWAvgLayerQuantizer
from heflp.training.params import (
flatten_model_params_per_layer,
unflatten_model_params_per_layer,
unflatten_model_params,
get_value_ranges_per_layer)
from heflp.training.runner import Runner
from heflp.utils import logger
from heflp.utils.perf import TimeMarker
import heflp
from heflp.secureproto.quantization.encode import decode_alpha_list, encode_all_layer_info

if heflp.info.EVALUATION_MODE:
    LOGGER_EVAL = logger.getEvalLogger()
T_MARKER = TimeMarker()
LOGGER = logger.getLogger()

def _last10_values(param):
    return list(param.values())[0][-10:]

class Flashev2Client(fl.client.Client):
    def __init__(self, cypher:Flashev2Cypher, quantizer:LayerQuantizer, model, runner:Runner, fit_epochs:int=1) -> None:
        super().__init__()
        self.cypher = cypher
        self.quantizer: MWAvgLayerQuantizer = quantizer
        self.model = model
        self.runner = runner
        self.fit_epochs = fit_epochs

    def _extract_model(self, model_params:Parameters, mwavg_param_bytes: bytes):
        if mwavg_param_bytes:
            T_MARKER.mark("decryption_start")
            encrypted_ret = Flashev2CTxt.from_bytes(model_params.tensors[0])
            q_ret = self.cypher.decrypt(encrypted_ret)
            T_MARKER.mark("decryption_done")
            T_MARKER.mark("dequantization_start")
            ret = self.quantizer.dequantize(q_ret, MWAvgParams.from_bytes(mwavg_param_bytes))
            T_MARKER.mark("dequantization_done")
            unflatten_model_params_per_layer(ret, self.model)
            LOGGER.debug(f"Received global model param [-10:0]: {_last10_values(ret)}")
        else:
            LOGGER.info("Initialize model")
            ret = fl.common.parameters_to_ndarrays(model_params)[0]
            unflatten_model_params(ret, self.model)
            LOGGER.debug(f"Received global model param [-10:0]: {ret[-10:]}")

    def fit(self, ins: FitIns) -> FitRes:
        LOGGER.info("Fit start")
        self._extract_model(ins.parameters, ins.config.get(Flashev2Property.MWAVG_PARAMS.value, None))
        T_MARKER.reset() # Reset the time marker
        flashev2_param = Flashev2CypherParams.from_bytes(ins.config.get(Flashev2Property.FLASHEV2_PARAMS.value))
        self.cypher.prepare_mask(flashev2_param)
        T_MARKER.mark("training_start")
        self.runner.train(self.model, epochs=self.fit_epochs)
        T_MARKER.mark("training_done")
        n_i = self.runner.get_dataset_size('train')
        flattened_ret = flatten_model_params_per_layer(self.model)
        LOGGER.debug(f"Trained local model param [-10:0]: {_last10_values(flattened_ret)}")
        LOGGER.debug(f"Degree n: {n_i}")
        T_MARKER.mark("quantization_start")
        if Flashev2Property.ALPHA_LIST.value in ins.config:
            new_alpha_lst = decode_alpha_list(ins.config.get(Flashev2Property.ALPHA_LIST.value))
            self.quantizer.update_alpha_map(new_alpha_lst)
        m_prime = ins.config.get(Flashev2Property.M_PRIME.value, 1.)
        LOGGER.debug(f"M'={m_prime}")
        q_ret = self.quantizer.quantize(flattened_ret, n_i/m_prime)
        param: MWAvgParams = self.quantizer.pop_cached_param()
        T_MARKER.mark("quantization_done")
        T_MARKER.mark("encryption_start")
        encrypted_tensor = self.cypher.encrypt(q_ret, flashev2_param)
        encrypted_tensor_bytes = encrypted_tensor.to_bytes()
        T_MARKER.mark("encryption_done")
        LOGGER.info("Fit done")
        T_MARKER.mark("transmit")
        v_ranges = get_value_ranges_per_layer(flattened_ret)
        layer_info_bytes = encode_all_layer_info(v_ranges, self.quantizer.layer_sizes)
        return FitRes(
            Status(Code.OK, 'Flashev2 encrypted'),
            Parameters([encrypted_tensor_bytes], 'Flashv2'),
            n_i,
            {Flashev2Property.MWAVG_PARAMS.value: param.to_bytes(),
             Flashev2Property.LAYER_INFO.value: layer_info_bytes}    
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        T_MARKER.mark("received")
        LOGGER.info("Evaluate start")
        self._extract_model(ins.parameters, ins.config.get(Flashev2Property.MWAVG_PARAMS.value, None))
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