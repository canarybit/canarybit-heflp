from tqdm import tqdm
from typing import Callable, Any
from .base import Runner, RunnerException
from heflp.utils import logger

LOGGER = logger.getLogger()
try:
    import torch
    from torch.utils.data import DataLoader
    from torch import Tensor
except ImportError as e:
    LOGGER.error("Pytorch is not installed")

class PytorchRunner(Runner):
    '''
    Pytorch Runner for training or testing the model
    '''
    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
        device: str,
        criterion,
        optimizer,
        metric_fn: Callable[[Tensor, Any], float]
    ) -> None:
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric_fn = metric_fn

    def train(self, model: torch.nn.Module, epochs: int = 1):
        """Train the model on the training set."""
        model.to(self.device).train()
        for _ in range(epochs):
            for images, labels in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                self.criterion(
                    model(images.to(self.device)), labels.to(self.device)
                ).backward()
                self.optimizer.step()

    def test(self, model: torch.nn.Module):
        """Validate the model on the test set."""
        model.to(self.device).eval()
        metric, loss = 0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(self.testloader):
                outputs = model(images.to(self.device))
                labels = labels.to(self.device)
                loss += self.criterion(outputs, labels).item()
                metric += self.metric_fn(outputs, labels)
        accuracy = metric / len(self.testloader.dataset)
        return loss, accuracy

    def get_dataset_size(self, mode: str):
        if mode == "train":
            return len(self.trainloader.dataset)
        elif mode == "test":
            return len(self.testloader.dataset)
        else:
            raise RunnerException("Dataset mode must be train or test")
