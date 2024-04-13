from typing import Generator, Tuple, Any, List
import random
import time
from .base import Runner, RunnerException


def static_weight_generator(weight:Any):
    '''Generator for static weight. Only for testing'''
    while True:
        yield weight

def randint_weight_generator(range_a:int, range_b:int):
    '''Generator for integer weight. Only for testing'''
    while True:
        yield random.randint(range_a, range_b)

def randfloat_weight_generator(range_a:float, range_b:float):
    '''Generator for float weight. Only for testing'''
    while True:
        yield random.uniform(range_a, range_b)


def loopint_weight_generator(looplist: List[Any]):
    '''Looply return the size value in the looplist. Only for testing'''
    while True:
        for v in looplist:
            yield v


class FakeRunner(Runner):
    '''For test only'''
    def __init__(self, 
                 train_size_gen: Generator, 
                 test_size_gen: Generator, 
                 train_time_s:float=3, 
                 test_time_s:float=3) -> None:
        self.train_size_gen = train_size_gen
        self.test_size_gen = test_size_gen
        self.train_time_s = train_time_s
        self.test_time_s = test_time_s

    def train(self, net, epochs: int = 1):
        time.sleep(self.train_time_s) # fake training

    def test(self, net)->Tuple[float, float]:
        time.sleep(self.test_time_s)  # fake training
        return 0.5, random.uniform(0.5, 1)

    def get_dataset_size(self, mode):
        if mode == "train":
            return next(self.train_size_gen)
        elif mode == "test":
            return next(self.test_size_gen)
        else:
            raise RunnerException("Dataset mode must be train or test")
