import flwr as fl

from heflp.secureproto.homoencrypschemes.flashe import FlasheCypher, FlasheCypherParams
from heflp.secureproto.quantization.quantizer import Quantizer
from heflp.secureproto.quantization.mwavg import MWAvgParams, MWAvgQuantizer
from heflp.training.params import flatten_model_params, unflatten_model_params
from heflp.training.runner import Runner
from heflp.utils import logger
from heflp.utils.perf import TimeMarker
import heflp

if heflp.info.EVALUATION_MODE:
    LOGGER_EVAL = logger.getEvalLogger()
T_MARKER = TimeMarker()
LOGGER = logger.getLogger()

class FlasheClient(fl.client.NumPyClient):
    def __init__(self, cypher:FlasheCypher, quantizer:Quantizer, model, runner:Runner, fit_epochs:int = 1) -> None:
        super().__init__()
        self.cypher = cypher
        self.quantizer = quantizer
        self.model = model
        self.runner = runner
        self.fit_epochs = fit_epochs

    def fit(self, parameters, config):
        LOGGER.info("Fit start")
        model, cypher, quantizer = self.model, self.cypher, self.quantizer
        LOGGER.debug(config)
        flashe_decrypt_params = FlasheCypherParams.create_by_str(config['decrypt'])
        decrypted_ret = cypher.decrypt(parameters[0], flashe_decrypt_params)
        sum_weights = config['sum_weights']
        if sum_weights:
            ret = quantizer.dequantize(decrypted_ret, sum_weights)
        else:
            ret = decrypted_ret
        LOGGER.debug(f"Received global model param [-10:0]: {ret[-10:]}")
        unflatten_model_params(ret, model)
        T_MARKER.reset() # Reset the time marker
        T_MARKER.mark("training_start")
        self.runner.train(model, epochs=self.fit_epochs)
        T_MARKER.mark("training_done")
        flattened_ret = flatten_model_params(model)
        n_i = self.runner.get_dataset_size('train')
        LOGGER.debug(f"Trained local model param [-10:0]: {flattened_ret[-10:]}")
        LOGGER.debug(f"Degree n: {n_i}")
        T_MARKER.mark("quantization_start")
        quantized_ret = quantizer.quantize(flattened_ret)
        T_MARKER.mark("quantization_done")
        T_MARKER.mark("encryption_start")
        flashe_encrypt_params = FlasheCypherParams.create_by_str(config['encrypt'])
        encrypted_ret = cypher.encrypt(quantized_ret*n_i, flashe_encrypt_params)
        T_MARKER.mark("encryption_done")
        LOGGER.debug(f"Encrypted local model param [-10:0]: {encrypted_ret[-10:]}")
        LOGGER.info("Fit done")
        T_MARKER.mark("transmit")
        return [encrypted_ret], n_i, {}

    def evaluate(self, parameters, config):
        LOGGER.info("Evaluate start")
        T_MARKER.mark("received")
        model, cypher, quantizer = self.model, self.cypher, self.quantizer
        T_MARKER.mark("decryption_start")
        flashe_decrypt_params = FlasheCypherParams.create_by_str(config['decrypt'])
        decrypted_ret = cypher.decrypt(parameters[0], flashe_decrypt_params)
        sum_weights = config['sum_weights']
        T_MARKER.mark("decryption_done")
        T_MARKER.mark("dequantization_start")
        if sum_weights:
            ret = quantizer.dequantize(decrypted_ret, sum_weights)
        else:
            ret = decrypted_ret
        T_MARKER.mark("dequantization_done")
        if heflp.info.EVALUATION_MODE:
            LOGGER_EVAL.debug(f"Time overhead | {T_MARKER.get_all_intervals()}") # Log the intervals for perf evaluation
        LOGGER.debug(f"Received global model param [-10:0]: {ret[-10:]}")
        unflatten_model_params(ret, model)
        loss, accuracy = self.runner.test(model)
        LOGGER.info("Evaluate done")
        return loss, self.runner.get_dataset_size('test'), {"accuracy": accuracy}