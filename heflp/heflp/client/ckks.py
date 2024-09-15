
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Status, Code, Parameters
import flwr as fl

from heflp.strategy.ckks import CKKSProperty
from heflp.secureproto.homoencrypschemes.ckks import CKKSCypher
from heflp.training.params import flatten_model_params, unflatten_model_params
from heflp.training.runner import Runner
from heflp.utils import logger
from heflp.utils.perf import TimeMarker
import heflp

if heflp.info.EVALUATION_MODE:
    LOGGER_EVAL = logger.getEvalLogger()
T_MARKER = TimeMarker()
LOGGER = logger.getLogger()

class CKKSClient(fl.client.Client):
    def __init__(self, cypher:CKKSCypher, model, runner:Runner, fit_epochs:int=1) -> None:
        super().__init__()
        self.cypher = cypher
        self.model = model
        self.runner = runner
        self.max_length = len(flatten_model_params(model))
        self.fit_epochs = fit_epochs

    def _extract_model(self, model_params:Parameters, initial):
        if not initial:
            T_MARKER.mark("decryption_start")
            encrypted_ret = self.cypher.from_bytes(model_params.tensors)
            ret = self.cypher.decrypt(encrypted_ret, self.max_length)
            T_MARKER.mark("decryption_done")
            T_MARKER.mark("dequantization_start")
            T_MARKER.mark("dequantization_done")
        else:
            LOGGER.info("Initialize model")
            ret = fl.common.parameters_to_ndarrays(model_params)[0]
        LOGGER.debug(f"Received global model param [-10:0]: {ret[-10:]}")
        unflatten_model_params(ret, self.model)

    def fit(self, ins: FitIns) -> FitRes:
        LOGGER.info("Fit start")
        if CKKSProperty.HE_CONTEXT.value in ins.config:
            self.cypher.set_context_bytes(ins.config.get(CKKSProperty.HE_CONTEXT.value))
        self._extract_model(ins.parameters, ins.config.get(CKKSProperty.INITIAL.value, None))
        T_MARKER.reset() # Reset the time marker
        T_MARKER.mark("training_start")
        self.runner.train(self.model, epochs=self.fit_epochs)
        T_MARKER.mark("training_done")
        flattened_ret = flatten_model_params(self.model)
        LOGGER.debug(f"Trained local model param [-10:0]: {flattened_ret[-10:]}")
        LOGGER.debug(f"Degree n: {self.runner.get_dataset_size('train')}")
        T_MARKER.mark("quantization_start")
        T_MARKER.mark("quantization_done")
        T_MARKER.mark("encryption_start")
        encrypted_tensor = self.cypher.encrypt(flattened_ret)
        encrypted_tensor_bytes = self.cypher.to_bytes(encrypted_tensor)
        T_MARKER.mark("encryption_done")
        LOGGER.info("Fit done")
        T_MARKER.mark("transmit")
        return FitRes(
            Status(Code.OK, 'CKKS encrypted'),
            Parameters(encrypted_tensor_bytes, 'CKKS'),
            self.runner.get_dataset_size('train'),
            {}    
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        T_MARKER.mark("received")
        LOGGER.info("Evaluate start")
        if CKKSProperty.HE_CONTEXT.value in ins.config:
            self.cypher.set_context_bytes(ins.config.get(CKKSProperty.HE_CONTEXT.value))
        self._extract_model(ins.parameters, ins.config.get(CKKSProperty.INITIAL.value, None))
        if heflp.info.EVALUATION_MODE:
            LOGGER_EVAL.debug(f"Time overhead | {T_MARKER.get_all_intervals()}") # Log the intervals for perf evaluation
        loss, accuracy = self.runner.test(self.model)
        LOGGER.info("Evaluate done | accuracy:", accuracy)
        return EvaluateRes(
            Status(Code.OK, 'success'),
            loss,
            self.runner.get_dataset_size('test'),
            {"accuracy": accuracy}
        )