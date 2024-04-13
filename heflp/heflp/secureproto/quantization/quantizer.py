import numpy as np
from typing import Iterable
from numpy.typing import NDArray
from heflp.typing import AlphaMap, LayerSizes, Layers
from collections import OrderedDict

class QuantizationException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def _static_quantize(value:NDArray[np.float32], r_max:float, int_bits:int):
    # first clipping
    value = np.clip(value, -r_max, r_max)

    # then quantize
    sign = np.sign(value)
    unsigned_value = value * sign
    unsigned_value = unsigned_value * (pow(2, int_bits - 1) - 1.0) / r_max
    value = unsigned_value * sign

    # then stochastic round
    size = value.shape
    value = np.floor(value + np.random.random(size)) # float to int
    # value = value.astype(object) # np.int to int (to avoid being tranferred to float later)

    # finally true value to 2's complement representation
    # value = true_to_two(value, int_bits)

    return value.astype(np.int32)

def _static_dequantize(value:NDArray[np.int32], r_max:float, int_bits:int):

    # 2's complement representation to true value
    # value = two_to_true(value, int_bits)

    # then dequantize
    sign = np.sign(value)
    unsigned_value = value * sign
    unsigned_value = unsigned_value * r_max / (pow(2, int_bits - 1) - 1.0)
    value = unsigned_value * sign

    return value.astype(np.float32)

class LayerQuantizer:
    def __init__(self, layer_sizes:LayerSizes, alpha_map:AlphaMap, bit_width=16) -> None:
        '''
        alpha_map: OrderedDict[layer_name(str), alpha(float)]
        layer_sizes: OrderedDict[layer_name(str), parameter number of this layer]

        Quantizer will quantize layer i using alpha i, from [-alpha, alpha] to [-2^15, 2^15] (if bit_width=16)
        '''
        self.bit_width = bit_width
        self.alpha_map = alpha_map
        self.layer_sizes = layer_sizes

    @classmethod
    def create(cls, layer_sizes:LayerSizes, default_alpha:float=1., bit_width=16)->"LayerQuantizer":
        alpha_map = OrderedDict()
        for layer in layer_sizes.keys():
            alpha_map[layer] = default_alpha
        return cls(layer_sizes, alpha_map, bit_width)

    def update_alpha_map(self, alpha_lst:Iterable[float]):
        '''Update alpha_map with new alphas in the list by order'''
        assert len(self.alpha_map) == len(alpha_lst)
        for layer_name, new_alpha in zip(self.alpha_map.keys(), alpha_lst):
            self.alpha_map[layer_name] = new_alpha

    def quantize(self, v:Layers, d=1.) -> NDArray[np.int32]:
        '''
        quantized rst = quantize(v*d) by layer
        return a flattened 1D vector
        '''
        try:
            rst_lst = []
            for name, p in v.items():
                assert len(p) == self.layer_sizes[name], f"{name}: {len(p)} != {self.layer_sizes[name]}"
                rst_lst.append(_static_quantize(p*d, self.alpha_map[name], self.bit_width))
            rst = np.concatenate(rst_lst)
            return rst
        except Exception as e:
            raise QuantizationException(f"Quantizing Failed: {e}")
    
    def dequantize(self, v:NDArray[np.int32], d=1.) -> Layers:
        '''
        plain rst = dequantize(v/d) by layer
        v: 1D array (will automatically detect layers)
        '''
        try:
            rst_dict = {}
            start = 0
            for name, sz in self.layer_sizes.items():
                v_layer = v[start:start+sz]
                rst_dict[name] = _static_dequantize(v_layer/d, self.alpha_map[name], self.bit_width)
                start += sz
            return rst_dict
        except Exception as e:
            raise QuantizationException(f"Unquantizing Failed: {e}")
        
    # def set_alpha_map(self, alpha_map:Dict[str, float]):
    #     self.alpha_map = alpha_map

class Quantizer:
    def __init__(self, r_max=1, bit_width=16) -> None:
        self.bit_width = bit_width
        self.r_max = r_max
    
    def quantize(self, v:NDArray, d=1.) -> NDArray:
        '''quantized rst = quantize(v*d)'''
        try:
            return _static_quantize(v*d, self.r_max, self.bit_width)
        except Exception as e:
            raise QuantizationException(f"Quantizing Failed: {e.args[0]}")
    
    def dequantize(self, v:NDArray, d=1.) -> NDArray:
        '''plain rst = dequantize(v/d)'''
        try:
            return _static_dequantize(v/d, self.r_max, self.bit_width)
        except Exception as e:
            raise QuantizationException(f"Unquantizing Failed: {e.args[0]}")
        
    def set_r_max(self, r_max:int=1):
        self.r_max = r_max
    

# Only for simple test
if __name__ == '__main__':
    x = np.array([-0.2, -0.6, 1.2, 0.1, 0.2])
    y = np.array([0.1, -0.6, 0, 0.1, -0.2])
    print(np.std(x))
    int_bits = 8
    quantized_value = _static_quantize(x, 0.5, int_bits)
    quantized_value2 = _static_quantize(y, 0.5, int_bits)
    print(quantized_value)
    print(quantized_value2)
    dequantized_value = _static_dequantize(quantized_value+quantized_value2, 0.5, int_bits)
    print(dequantized_value)
    print(x+y)

    quantizer = Quantizer(1, 16)
    qx = quantizer.quantize(x)
    print(f"{x} -> \n{qx} -> \n{quantizer.dequantize(qx)}")