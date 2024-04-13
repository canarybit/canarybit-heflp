import struct
from typing import List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from heflp.typing import LayerVRanges, LayerSizes

def encode_alpha_list(alpha_lst:List[float]):
    return struct.pack("<"+"f"*len(alpha_lst), *alpha_lst)

def decode_alpha_list(alpha_lst_bytes:bytes)->List[float]:
    return list(struct.unpack("<"+"f"*(len(alpha_lst_bytes)//4), alpha_lst_bytes))

def encode_layer_sizes(layer_sizes:LayerSizes):
    layer_sz_lst = list(layer_sizes.values())
    return struct.pack("<"+"I"*len(layer_sz_lst), *layer_sz_lst)

def decode_layer_sizes(layer_sz_bytes:bytes)->Tuple[int]:
    return struct.unpack("<"+"I"*(len(layer_sz_bytes)//4), layer_sz_bytes)

def encode_value_ranges(ranges:LayerVRanges):
    values = np.concatenate(list(ranges.values()))
    return struct.pack("<"+"f"*len(values), *values)

def decode_value_ranges(ranges_bytes:bytes)->NDArray:
    '''
    Return an 2D Array:
    [Layer1 range([max, min]), Layer2 range, Layer3 range...]
    '''
    ranges_list = struct.unpack("<"+"f"*(len(ranges_bytes)//4), ranges_bytes)
    return np.reshape(ranges_list, newshape=(-1, 2))

def encode_all_layer_info(value_ranges:LayerVRanges, layer_sizes:LayerSizes):
    assert len(value_ranges) == len(layer_sizes)
    byte_arr1 = encode_value_ranges(value_ranges)
    byte_arr2 = encode_layer_sizes(layer_sizes)
    return byte_arr1 + byte_arr2

def decode_all_layer_info(byte_arr:bytes):
    l = len(byte_arr) // 3
    value_ranges = decode_value_ranges(byte_arr[:2*l])
    layer_sizes = decode_layer_sizes(byte_arr[2*l:])
    return value_ranges, layer_sizes