import numpy as np
from numpy.typing import NDArray
import struct
from typing import List, Union, Dict
from dataclasses import dataclass
from collections import OrderedDict
from heflp.typing import LayerSizes, AlphaMap
from heflp.secureproto.quantization.quantizer import Quantizer, QuantizationException, LayerQuantizer


# The FL process follows this procedure: 
# Assume:
#   the weights of client i = w_i (list of float)
#   total num of samples = n_i (int)
# Step 1 (client): tmp1:list of float = w * ( n_i / m_i ) where m_i = smallest 2^N that is larger than n_i, i.e. n=900, m=1024
# Step 2 (client): tmp2:list of int = quantize(tmp1)
# Step 3 (client): tmp3 = encrypt(tmp2)
# ======================================
# step 4 (server): collect all tmp3_i and:
#   calculate M = min(m_i)  => to make sure for any i, m_i / m is integer
#   for all i: tmp4_i = tmp3_i * (m_i / M)
#   tmp4 = sum(tmp4_i)
#   N = sum(n_i)
# ======================================
# step 5 (client): tmp5:list of int = decrypt(tmp4)
# step 6 (client): tmp6:list of float = tmp5 * (M / N)
# step 7 (client): tmp7:list of int = convert_to_approximate_int(tmp6), i.e. 6.4=>6, 6.8=>7
# step 8 (client): w_new:list of float = dequantize(tmp7)

@dataclass
class MWAvgParams():
    degree: float
    m: float

    def get_d_div_m(self):
        '''Get the result of degree/m'''
        return self.degree/self.m
    
    def to_bytes(self) -> bytes:
        pack_bytes = struct.pack('<ff', self.degree, self.m)  # 'i' denotes integer format (4 bytes)
        return pack_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> 'MWAvgParams':
        degree, m = struct.unpack('<ff', data)
        return cls(degree, m)
    
    @classmethod
    def combine(cls, param_list: List['MWAvgParams']) -> 'MWAvgParams':
        '''Combine a list of param, where M = min(m_i), D = sum(d_i)'''
        M = min([param.m for param in param_list])
        d_sum = sum([param.degree for param in param_list])
        return cls(d_sum, M)
    
    def calculate_miM_list(self, param_list: List['MWAvgParams']) -> List[int]:
        '''This method can only be used for combined MWAvgParam object, get list of m_i/M'''
        return [int(param.m // self.m) for param in param_list]

# Calculate the first degree: 
def _calculate_m(s:Union[float, int]) -> float:
    p = np.ceil(np.log2(s))
    m = 2**p
    return max(m, 1.)
    
class MWAvgLayerQuantizer(LayerQuantizer):
    def __init__(self, layer_sizes:LayerSizes, alpha_map=AlphaMap, bit_width=16) -> None:
        super().__init__(layer_sizes, alpha_map, bit_width)
        self.param_tmp = None
    
    @classmethod
    def create(cls, layer_sizes:LayerSizes, default_alpha:float=1., bit_width=16)->"MWAvgLayerQuantizer":
        alpha_map = OrderedDict()
        for layer in layer_sizes.keys():
            alpha_map[layer] = default_alpha
        return cls(layer_sizes, alpha_map, bit_width)

    def quantize(self, v:Dict[str, NDArray[np.float32]], d=1.):
        '''rst = quantize(v, d/m)'''
        m = _calculate_m(d)
        self.param_tmp = MWAvgParams(d, m)
        return super().quantize(v, d/m)
    
    def pop_cached_param(self) -> MWAvgParams:
        if self.param_tmp:
            tmp = self.param_tmp
            self.param_tmp = None
        else:
            raise QuantizationException("Quantizer param cache is empty, please first quantize!")
        return tmp
    
    def dequantize(self, v: NDArray[np.int32], param: Union[MWAvgParams, float]=1.):
        if isinstance(param, MWAvgParams):
            dm = param.get_d_div_m()
        else:
            dm = param
        return super().dequantize(v, dm)


class MWAvgQuantizer(Quantizer):
    def __init__(self, r_max=1, bit_width=16) -> None:
        super().__init__(r_max, bit_width)
        self.param_tmp = None
    
    def quantize(self, v: NDArray, d=1.) -> NDArray:
        '''rst = quantize(v, d/m)'''
        m = _calculate_m(d)
        self.param_tmp = MWAvgParams(d, m)
        return super().quantize(v, d/m)
    
    def pop_cached_param(self) -> MWAvgParams:
        if self.param_tmp:
            tmp = self.param_tmp
            self.param_tmp = None
        else:
            raise QuantizationException("Quantizer param cache is empty, please first quantize!")
        return tmp
    
    def dequantize(self, v: NDArray, param: Union[MWAvgParams, float]=1.) -> NDArray:
        if isinstance(param, MWAvgParams):
            dm = param.get_d_div_m()
        else:
            dm = param
        return super().dequantize(v, dm)
    

if __name__ == '__main__':
    x1 = np.array([0.2, 0.4])
    x2 = np.array([0.4, 0.8])
    quantizer = MWAvgQuantizer(bit_width=20)
    # param1 = quantizer.pop_cached_param()
    qtz_x1 = quantizer.quantize(x1, 2)
    param1 = quantizer.pop_cached_param()
    param1_bytes = param1.to_bytes()
    param1 = MWAvgParams.from_bytes(param1_bytes)
    qtz_x2 = quantizer.quantize(x2, 1)
    param2 = quantizer.pop_cached_param()
    
    param3 = MWAvgParams.combine([param1, param2])
    miM_list = param3.calculate_miM_list([param1, param2])
    qtz_x3 = qtz_x1*miM_list[0] + qtz_x2*miM_list[1]

    rst = quantizer.dequantize(qtz_x3, param3.get_d_div_m())
    print(rst*3)
    print(qtz_x1.dtype)
