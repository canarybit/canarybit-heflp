from typing import Iterable, Tuple
from ..common.aciq import get_alpha_gaus
import numpy as np

def decide_alpha_per_layer(vranges:Iterable[Tuple[np.float32]], layer_sizes:Iterable[int]):
    assert len(vranges) == len(layer_sizes)
    alpha_lst = [get_alpha_gaus(vrange, layer_sz)  for vrange, layer_sz in zip(vranges, layer_sizes)]
    return alpha_lst