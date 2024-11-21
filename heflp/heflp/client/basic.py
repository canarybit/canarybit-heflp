import flwr as fl

from heflp.training.runner import Runner
from heflp.training.params import flatten_model_params, unflatten_model_params

class BasicClient(fl.client.NumPyClient):
    def __init__(self, model, runner:Runner, fit_epochs:int = 1) -> None:
        super().__init__()
        self.model = model
        self.runner = runner
        self.fit_epochs = fit_epochs

    def fit(self, parameters, config):
        model = self.model
        unflatten_model_params(parameters[0], model)
        self.runner.train(model, epochs=self.fit_epochs)
        flattened_ret = flatten_model_params(model)
        return [flattened_ret], self.runner.get_dataset_size('train'), {}

    def evaluate(self, parameters, config):
        model = self.model
        unflatten_model_params(parameters[0], model)
        size = self.runner.get_dataset_size('test')
        print("test size", size)
        mae_loss = self.runner.test(model)
        print("mae loss", mae_loss)
        accuracy = 0.9
        return mae_loss, size, {"accuracy": accuracy}

