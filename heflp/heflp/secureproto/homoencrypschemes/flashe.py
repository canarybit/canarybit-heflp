from typing import List, Tuple, Dict, Mapping, Union

from heflp.secureproto.common.aes_prp import PsuedoRandomPermutation
from heflp.secureproto.homoencrypschemes.cypher import CypherBase, HomoEncrypException
import numpy as np
import os
import json

def _serialize_dict(dictionary:Dict[bytes, float]):
    serialized_dict = {}
    for key, value in dictionary.items():
        serialized_key = key.decode('utf-8')  # Convert bytes to string
        serialized_dict[serialized_key] = value
    return json.dumps(serialized_dict)

def _deserialize_dict(serialized_string:str):
    serialized_dict = json.loads(serialized_string)
    deserialized_dict = {}
    for serialized_key, value in serialized_dict.items():
        deserialized_key = serialized_key.encode('utf-8')  # Convert string to bytes
        deserialized_dict[deserialized_key] = value
    return deserialized_dict

class FlasheCypherParams(object):
    def __init__(self, begin:int, end:int, index_prefix_dir:Mapping[bytes, float]):
        self.begin = begin
        self.end = end
        self.index_prefix_dir = index_prefix_dir

    @classmethod
    def create(cls, begin:int=0, end:int=0, index_prefix_dir:Mapping[bytes, float]={}):
        return FlasheCypherParams(begin, end, index_prefix_dir)

    @classmethod
    def create_by_str(cls, serialized_str:str):
        return FlasheCypherParams(*cls.deserialize(serialized_str))

    def get_params(self):
        return self.begin, self.end, self.index_prefix_dir

    # self + b * m
    def multi_add(self, b:Union[Mapping[bytes, float], 'FlasheCypherParams'], m:float):
        if isinstance(b, FlasheCypherParams):
            b = b.index_prefix_dir
        combined_dict = dict(self.index_prefix_dir)
        for key, value in b.items():
            if key in combined_dict:
                combined_dict[key] += value*m
                if combined_dict[key] == 0:
                    combined_dict.pop(key)
            else:
                combined_dict[key] = value*m
        return combined_dict

    def multi_add_update(self, b:Union[Mapping[bytes, float], 'FlasheCypherParams'], m: float):
        self.index_prefix_dir = self.multi_add(b, m)
        return self.index_prefix_dir

    def serialize(self):
        serialized_idx_prefix_dir = _serialize_dict(self.index_prefix_dir)
        return f"{self.begin} {self.end} {serialized_idx_prefix_dir}"
    
    @classmethod
    def deserialize(cls, serialized_str:str):
        params = serialized_str.split(" ", 2)
        begin = int(params[0])
        end = int(params[1])
        idx_prefix_dir = _deserialize_dict(params[2])
        return begin, end, idx_prefix_dir
        
def _static_prepare_mask_single(begin, end, prp_seed,
                            int_bits, index_prefix_for_add):
    local_prp = PsuedoRandomPermutation()
    local_prp.generate_key(assigned_key=prp_seed)
    return _prepare_mask_single(local_prp, int_bits, begin, end, index_prefix_for_add)

def _prepare_mask_single(prp:PsuedoRandomPermutation, int_bits, begin, end, index_prefix):
    length = end - begin
    merge_size = 128 // int_bits
    merge_num = (length - 1) // merge_size + 1
    add_terms = []

    for i in range(merge_num):
        b = i * merge_size
        e = min((i+1) * merge_size, length)

        # i + begin can guarantee to be unique globally
        index_for_add  =  index_prefix  + (i  +  begin).to_bytes(8, 'big')

        add_term_bytes  =  prp.get_permutation(index_for_add)
        add_term_s  =  int.from_bytes(add_term_bytes, 'big')

        mask = (1 << int_bits) - 1
        for j in range(b, e):
            add_term = add_term_s & mask
            add_terms.append(add_term)
            add_term_s >>= int_bits

    return np.array(add_terms)


def _static_prepare_mask_multiple(begin, end, prp_seed, int_bits, index_prefix_dir):
    local_prp = PsuedoRandomPermutation()
    local_prp.generate_key(assigned_key=prp_seed)
    return _prepare_mask_multiple(local_prp, int_bits, FlasheCypherParams.create(begin, end, index_prefix_dir))

def _create_term_s(prefix: bytes, sd: bytes, prp: PsuedoRandomPermutation):
    term_bytes = prp.get_permutation(prefix + sd)
    return int.from_bytes(term_bytes, 'big')

def _prepare_mask_multiple(prp:PsuedoRandomPermutation, int_bits, params:FlasheCypherParams):
    begin, end, index_prefix_dir = params.get_params()
    length = end - begin
    merge_size = 128 // int_bits
    merge_num = (length - 1) // merge_size + 1
    terms = []

    for i in range(merge_num):
        b = i * merge_size
        e = min((i + 1) * merge_size, length)

        # i + begin can guarantee to be unique globally
        sd:bytes = (i + begin).to_bytes(8, 'big')

        term_s_list = [_create_term_s(k[0], sd, prp) 
                           for k in index_prefix_dir.items()]

        mask = (1 << int_bits) - 1
        for j in range(b, e):
            term = 0

            for k_idx, k in enumerate(index_prefix_dir.items()):
                term += (term_s_list[k_idx] & mask) * k[1]
                term_s_list[k_idx] >>= int_bits

            terms.append(term)

    return np.array(terms, dtype=np.int32)

class FlasheCypher(CypherBase):
    def __init__(self, seed:bytes, bit_width=16) -> None:
        super().__init__(seed)
        self.prp = PsuedoRandomPermutation()
        self.prp.generate_key(assigned_key=self.seed)
        self.bit_width = bit_width

    @classmethod
    def create_with_autoseed(cls, bit_width=16):
        prp_seed_len = 256
        BITS_PER_BYTE = 8
        prp_seed = os.urandom(prp_seed_len//BITS_PER_BYTE)
        return FlasheCypher(prp_seed, bit_width)
    
    def get_seed(self):
        return self.seed

    def encrypt(self, plaintext: np.ndarray, params: FlasheCypherParams):
        try:
            mask = _prepare_mask_multiple(self.prp, self.bit_width, params)
            return plaintext + mask
        except Exception as e:
            raise HomoEncrypException(f'Flashe Encryption Failed: {e.args[0]}')
        
    @classmethod
    def static_encrypt(cls, plaintext, begin, end, prp_seed, bit_width, index_prefix_dir):
        try:
            mask = _static_prepare_mask_multiple(begin, end, prp_seed, bit_width, index_prefix_dir)
            return plaintext + mask
        except Exception as e:
            raise HomoEncrypException(f'Flashe Encryption Failed: {e.args[0]}')

    def decrypt(self, cyphertext: np.ndarray, params: FlasheCypherParams):
        try:
            unmask = _prepare_mask_multiple(self.prp, self.bit_width, params)
            return cyphertext - unmask
        except Exception as e:
            raise HomoEncrypException(f'Flashe Decryption Failed: {e.args[0]}')
        
    @classmethod
    def static_decrypt(cls, cyphertext, begin, end, prp_seed, bit_width, index_prefix_dir):
        try:
            mask = _static_prepare_mask_multiple(begin, end, prp_seed, bit_width, index_prefix_dir)
            return cyphertext - mask
        except Exception as e:
            raise HomoEncrypException(f'Flashe Decryption Failed: {e.args[0]}')

    @classmethod
    def concat_prefix(cls, i:int, j:int):
        return i.to_bytes(4, 'big') + j.to_bytes(4, 'big')

if __name__ == "__main__":
    print("test start")

    prp_seed_len = 256
    BITS_PER_BYTE = 8
    prp_seed = os.urandom(prp_seed_len//BITS_PER_BYTE)

    idx_prefix_for_add = FlasheCypher.concat_prefix(1,2)
    idx_prefix_for_add2 = FlasheCypher.concat_prefix(2,2)
    print(f"idx_prefix_for_add: {idx_prefix_for_add}")

    add_items = _static_prepare_mask_single(0, 10, prp_seed, 32, idx_prefix_for_add)
    print(add_items)
    # print(type(add_items[0]))

    minus_items = _static_prepare_mask_multiple(0, 10, prp_seed, 32, {idx_prefix_for_add:-0.5})
    print(minus_items)

    print(0.5*add_items + minus_items)

    





