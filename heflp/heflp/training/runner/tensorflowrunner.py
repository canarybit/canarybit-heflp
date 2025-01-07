from typing import Generator
from .base import Runner, RunnerException
from heflp.utils import logger
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

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
        print("tsTrainingxxxxxxxxxxxxxx")
        self._compile_model(model)

        early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True,
                                   min_delta=0.001, mode='min')
        model.fit(x=self.train_gen, steps_per_epoch=self.batch_size, epochs=epochs,callbacks=[early_stopping])

        X_train = self.X_train.astype('float32')
        pred_train = model.predict(X_train)
        absolute_errors = np.abs(X_train - pred_train)
        mask = (X_train != -1.0).astype(np.float32)
        absolute_errors[X_train == -1.0] = 0.0
        mae = np.sum(absolute_errors, axis=(1, 2)) / np.sum(mask, axis=(1, 2))
        X_train_df=pd.DataFrame()
        X_train_df["error"] = mae

        threshold=np.percentile(X_train_df.error, 99)
        print("THRESHOLD :", threshold)

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