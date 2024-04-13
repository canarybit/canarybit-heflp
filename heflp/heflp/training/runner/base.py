from typing import Tuple

class RunnerException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Runner(object):
    def train(self, model, epochs: int = 1):
        pass

    def test(self, model)->Tuple[float, float]:
        pass

    def get_dataset_size(self, mode:str):
        "Get dataset size, mode = train | test"
        pass