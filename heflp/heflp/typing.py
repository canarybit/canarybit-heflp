from typing import Dict, Tuple, OrderedDict, Any
from numpy.typing import NDArray
import numpy as np

Layers = Dict[str, NDArray[np.float32]]
LayerVRanges = Dict[str, Tuple[float, float]]
AlphaMap = OrderedDict[str, float]
LayerSizes = OrderedDict[str, int]
Vector1D = NDArray[np.float32]
MLModel = Any