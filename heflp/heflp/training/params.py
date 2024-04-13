import numpy as np
from numpy.typing import NDArray
from heflp.utils import logger
from heflp.typing import Layers, LayerVRanges, LayerSizes, MLModel
import sys
from .check import check_import
from typing import Tuple

# LOGGER = logger.getLogger()
# MLModel = Union[torch.nn.Module, keras.Model]

if check_import("torch"):
    import torch
# else:
#     LOGGER.warning("Pytorch is not installed")
if check_import("keras"):
    import keras
# else:
#     LOGGER.warning("Tensorflow is not installed")

def get_value_ranges_per_layer(layers:Layers)->LayerVRanges:
    v_range = {}
    for layer, param in layers.items():
        v_range[layer] = (np.max(param), np.min(param))
    return v_range

def get_layer_sizes(layers:Layers)->LayerSizes:
    layer_sizes = {}
    for layer, param in layers.items():
        layer_sizes[layer] = len(param)
    return layer_sizes

def flatten_model_params_per_layer(model:MLModel)->Layers:
    '''Flatten the model into a Dict[layer name (str), flattened layer params (ndarray)]'''
    params = {}
    if "torch" in sys.modules and isinstance(model, torch.nn.Module):
        try:
            for name, layer in model.named_children():
                if any(param.numel() > 0 for param in layer.parameters()):
                    params[name] = np.concatenate(
                        [param.detach().numpy().ravel()
                        for param in layer.parameters()]
                    )
        except Exception as e:
            # LOGGER.error("Failed to Flatten model parameters")
            raise e
    elif "keras" in sys.modules and isinstance(model,keras.Model):
        try:
            for layer in model.layers:
                if len(layer.get_weights()):
                    params[layer.name] = np.concatenate(
                        [param.flatten() for param in layer.get_weights()]
                    )
        except Exception as e:
            # LOGGER.error("Failed to Flatten model parameters")
            raise e
    else:
        raise ValueError("Invalid model type. Expecting PyTorch or Keras model.")
    return params

def unflatten_model_params_per_layer(flattened_params:Layers, model:MLModel):
    '''
    Unflatten the params and update the model parameters accordingly
    flattened param is a Dict[layer name (str), flattened layer params (ndarray)]
    '''
    if "torch" in sys.modules and isinstance(model, torch.nn.Module):
        for name, layer in model.named_children():
            start = 0
            for param in layer.parameters():
                end = start + param.numel()
                param.data = torch.from_numpy(flattened_params[name][start:end].reshape(param.shape).astype(np.float32))
                start = end
    elif "keras" in sys.modules and isinstance(model, keras.Model):
        unflattened_weights = []
        for layer in model.layers:
            start = 0
            for param in layer.get_weights():
                shape = param.shape
                end = start + np.prod(shape)
                unflattened_weight = np.array(flattened_params[layer.name][start:end]).reshape(shape).astype(np.float32)
                unflattened_weights.append(unflattened_weight)
                start = end
        model.set_weights(unflattened_weights)
    else:
        raise ValueError("Invalid model type. Expecting PyTorch or Keras model.")
    return model

def flatten_model_params(model):
    '''Flatten the model into a 1D Numpy array (Vector)'''
    if "torch" in sys.modules and isinstance(model, torch.nn.Module):
        params = np.concatenate(
            [param.detach().numpy().flatten() 
             for param in model.parameters()]
        )
    elif "keras" in sys.modules and isinstance(model,keras.Model):
        params = np.concatenate(
            [param.flatten() for param in model.get_weights()]
        )
    else:
        raise ValueError("Invalid model type. Expecting PyTorch or Keras model.")
    return params

def unflatten_model_params(flattened_params:NDArray, model):
    '''Unflatten the 1D Vector and update the model parameters accordingly'''
    if "torch" in sys.modules and isinstance(model, torch.nn.Module):
        start = 0
        for param in model.parameters():
            end = start + param.numel()
            param.data = torch.from_numpy(flattened_params[start:end].reshape(param.shape).astype(np.float32))
            start = end
    elif "keras" in sys.modules and isinstance(model, keras.Model):
        unflattened_weights = []
        start = 0
        for param in model.get_weights():
            shape = param.shape
            end = start + np.prod(shape)
            unflattened_weight = np.array(flattened_params[start:end]).reshape(shape).astype(np.float32)
            unflattened_weights.append(unflattened_weight)
            start = end
        model.set_weights(unflattened_weights)
    else:
        raise ValueError("Invalid model type. Expecting PyTorch or Keras model.")
    return model

def get_model_params_num(model):
    '''Get the number of model parameters'''
    if "torch" in sys.modules and isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters())
    elif "keras" in sys.modules and isinstance(model, keras.Model):
        return model.count_params()
    else:
        raise ValueError("Invalid model type. Expecting PyTorch or Keras model.")
    
def save_flattened_model_params(filepath:str, model):
    '''Save the model parameters as a 1D vector'''
    np.save(filepath, flatten_model_params(model))

# Only for test
if __name__ == '__main__':
    import torch
    import numpy as np
    import torch.nn as nn
    import torch.nn.init as init
    import random
    import sys
    sys.path.append(".")
    from src.models import Net

    @torch.no_grad()
    def init_weights(m):
    #   print(m)
      if type(m) == nn.Linear:
            m.weight.fill_(np.random.rand())
            # print(m.weight)

    # Define your model
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    # Create an instance of your model
    # model = MyModel()
    # model.apply(init_weights)
    model = Net()
    model.load_state_dict(torch.load("./outputs/tmp.pth"))

    # print(model.state_dict().keys())
    # conv1_weight_numpy_layer = model.state_dict()['fc1.bias'].cpu().numpy()
    # print(np.mean(conv1_weight_numpy_layer), np.max(conv1_weight_numpy_layer), np.min(conv1_weight_numpy_layer))
    print(model.state_dict()['fc1.bias'])
    # Flatten and convert the model parameters to NumPy arrays
    flattened_params = flatten_model_params(model)
    # Reshape and assign the flattened parameters back to the model
    model = unflatten_model_params(flattened_params, model)

    # Test the recovered model
    print(model.state_dict()['fc1.bias'])