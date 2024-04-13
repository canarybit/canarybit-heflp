from typing import Generator
from .base import Runner, RunnerException
from heflp.utils import logger

LOGGER = logger.getLogger()
try:
    import keras
except ImportError as e:
    LOGGER.error("Tensorflow is not installed")

class TensorflowRunner(Runner):
    '''
    Tensorflow Runner for training or testing the model
    '''
    def __init__(
        self,
        train_gen: Generator,
        test_gen: Generator,
        train_steps: int,
        test_steps: int,
        criterion: str,
        optimizer: keras.optimizers.Optimizer,
        metric: keras.metrics.Metric
    ) -> None:
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.train_steps = train_steps
        self.test_steps = test_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric = metric # Only support one metric now

    def _compile_model(self, model: keras.Model, force: bool=False):
        '''
        Compile the model, if force==False, only compile when the model is not compiled yet.
        ATTENTION: Recompiling could cause a leakage of memory!
        '''
        try:
            if model._is_compiled and not force:
                return
            model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=[self.metric])
        except Exception as e:
            raise RunnerException(f"Failed to compile the model: {e.args[0]}")

    def train(self, model: keras.Model, epochs: int = 1):
        self._compile_model(model)
        model.fit_generator(generator=self.train_gen, steps_per_epoch=self.train_steps, epochs=epochs)

    def test(self, model: keras.Model):
        self._compile_model(model)
        rst = model.evaluate_generator(generator=self.test_gen, steps=self.test_steps)
        return rst[0], rst[1] # loss, metric

    def get_dataset_size(self, mode: str):
        if mode == "train":
            return self.train_steps
        elif mode == "test":
            return self.test_steps
        else:
            raise RunnerException("Dataset mode must be train or test")