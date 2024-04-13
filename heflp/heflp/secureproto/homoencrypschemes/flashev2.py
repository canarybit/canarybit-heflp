from typing import List, Tuple, Dict, Mapping, Union

from heflp.secureproto.common.aes_prp import PsuedoRandomPermutation
from heflp.secureproto.homoencrypschemes.cypher import CypherBase, HomoEncrypException
import numpy as np
from numpy.typing import NDArray
import os
from dataclasses import dataclass
import pickle
from struct import Struct
from itertools import chain
import concurrent.futures

@dataclass
class Flashev2CypherParams:
    begin: int
    end: int
    S: Dict[bytes, int]
    length: int

    def __init__(self, begin:int, end:int, S:Dict[bytes, int]):
        if (begin > end):
            raise HomoEncrypException(f"Flashev2: begin={begin} should be smaller than end={end}")
        self.begin = begin
        self.end = end
        self.length = end - begin
        if isinstance(S, dict):
            self.S = S
        else:
            try:
                self.S = dict(S)
            except Exception as e:
                raise HomoEncrypException(e)

    def get_params(self):
        return self.begin, self.end, self.S, self.length

    def to_bytes(self):
        packed_params = (self.begin, self.end, self.S)
        return pickle.dumps(packed_params)

    @classmethod
    def from_bytes(cls, data:bytes):
        begin, end, S = pickle.loads(data)
        return cls(begin, end, S)

    @classmethod
    def concat_prefix(cls, i:int, j:int):
        return i.to_bytes(4, 'big') + j.to_bytes(4, 'big')


@dataclass
class Flashev2CTxt(Flashev2CypherParams):
    ciphertext: NDArray[np.int32]

    def __init__(self, begin:int, end:int, S:Dict[bytes, int], ciphertext:NDArray[np.int32]):
        super().__init__(begin, end, S)
        if ciphertext.shape != (self.length,): # 1D array with length of begin-end
            raise HomoEncrypException(
                f"shape of ciphertext={ciphertext.shape} does not match [(end - begin)]={(end-begin,)}")
        self.ciphertext = ciphertext

    @classmethod
    def create_by_params(cls, params:Flashev2CypherParams, ciphertext:NDArray[np.int32]):
        return Flashev2CTxt(params.begin, params.end, params.S, ciphertext)

    def __mul__(self, other: int):
        if not isinstance(other, int):
            raise HomoEncrypException(
                f"Flashev2 ciphertext only support integer multiplication. Invalid factor {other}({type(other)})")
        S = dict({k:v*other for k,v in self.S.items()})
        ciphertext = self.ciphertext * other
        return Flashev2CTxt(self.begin, self.end, S, ciphertext)

    def __add__(self, other: "Flashev2CTxt"):
        if other.begin != self.begin or other.end != self.end:
            raise HomoEncrypException(
                f"Flashev2: begin or end of two additive factors does not match: f{(self.begin,self.end)}+f{(other.begin,other.end)}")
        ciphertext = self.ciphertext + other.ciphertext
        S = self.S.copy()
        for key, value in other.S.items():
            if key in S:
                S[key] += value
                if S[key] == 0:
                    S.pop(key)
            else:
                S[key] = value
        return Flashev2CTxt(self.begin, self.end, S, ciphertext)
    
    def to_bytes(self):
        packed_params = (self.begin, self.end, self.S, self.ciphertext.tobytes())
        return pickle.dumps(packed_params)

    @classmethod
    def from_bytes(cls, data:bytes):
        begin, end, S, ciphtertext = pickle.loads(data)
        return cls(begin, end, S, np.frombuffer(ciphtertext, dtype=np.int32))

def _static_prepare_mask_multiple(begin, end, prp_seed, int_bits, index_prefix_dir):
    local_prp = PsuedoRandomPermutation()
    local_prp.generate_key(assigned_key=prp_seed)
    return _prepare_mask_multiple(local_prp, int_bits, Flashev2CypherParams(begin, end, index_prefix_dir))

def _create_term_s(prefix: bytes, sd: bytes, prp: PsuedoRandomPermutation):
    term_bytes = prp.get_permutation(prefix + sd)
    return int.from_bytes(term_bytes, 'big')

def _create_term_s_bytes(prefix: bytes, sd: bytes, prp: PsuedoRandomPermutation):
    return prp.get_permutation(prefix + sd)

def _prepare_mask_multiple(prp:PsuedoRandomPermutation, int_bits, params:Flashev2CypherParams):
    begin, end, index_prefix_dir, length = params.get_params()
    merge_size = 128 // int_bits
    merge_num = (length - 1) // merge_size + 1
    
    if int_bits == 16 or int_bits == 32 or int_bits == 64:  # Accelerate mask creating
        if int_bits == 16:
            stct = Struct('>hhhhhhhh')
        elif int_bits == 32:
            stct = Struct('>iiii')
        else:
            stct = Struct('>qq')
        
        dtype = getattr(np, f"int{int_bits*2}") # int_bits * 2 makes sure the terms results will not overflow
        terms = np.zeros(length, dtype=dtype)
        for k,v in index_prefix_dir.items():
            terms_k = []
            for i in range(merge_num):
                sd:bytes = (i + begin).to_bytes(8, 'big')
                term_s_bytes = _create_term_s_bytes(k, sd, prp)
                terms_k.append(stct.unpack(term_s_bytes))
            terms_k_list = list(chain.from_iterable(terms_k))
            terms += np.array(terms_k_list[:length], dtype=dtype) * v
        return terms
    else:
        terms = []

        for i in range(merge_num):
            b = i * merge_size
            e = min((i + 1) * merge_size, length)

            # i + begin can guarantee to be unique globally
            sd:bytes = (i + begin).to_bytes(8, 'big')

            term_s_list = [_create_term_s(k, sd, prp) 
                            for k,_ in index_prefix_dir.items()]

            mask = (1 << int_bits) - 1
            for j in range(b, e):
                term = 0

                for k_idx, (k,v) in enumerate(index_prefix_dir.items()):
                    term += (term_s_list[k_idx] & mask) * v
                    term_s_list[k_idx] >>= int_bits
                terms.append(term)

        return np.array(terms)

class Flashev2Cypher(CypherBase):
    def __init__(self, seed:bytes, bit_width=16) -> None:
        super().__init__(seed)
        self.prp = PsuedoRandomPermutation()
        self.prp.generate_key(assigned_key=self.seed)
        self.bit_width = bit_width
        self.mask_future: concurrent.futures.Future = None
        self.executor = concurrent.futures.ProcessPoolExecutor()

    @classmethod
    def create_with_autoseed(cls, bit_width=16):
        prp_seed_len = 256
        BITS_PER_BYTE = 8
        prp_seed = os.urandom(prp_seed_len//BITS_PER_BYTE)
        return Flashev2Cypher(prp_seed, bit_width)
    
    def get_seed(self):
        return self.seed

    def prepare_mask(self, params: Flashev2CypherParams):
        self.mask_future = self.executor.submit(_static_prepare_mask_multiple, params.begin, params.end, self.seed, self.bit_width, params.S)

    def encrypt(self, plaintext: NDArray, params: Flashev2CypherParams) -> Flashev2CTxt:
        try:
            # mask = _prepare_mask_multiple(self.prp, self.bit_width, params)
            if self.mask_future:  # Using mask precomputing
                mask = self.mask_future.result()
            else:     # Not using mask precomputing
                mask = _prepare_mask_multiple(self.prp, self.bit_width, params)
            return Flashev2CTxt.create_by_params(params, plaintext + mask)
        except Exception as e:
            raise HomoEncrypException(f'Flashe Encryption Failed: {e.args[0]}')
        
    @classmethod
    def static_encrypt(cls, plaintext, begin, end, prp_seed, bit_width, index_prefix_dir):
        try:
            mask = _static_prepare_mask_multiple(begin, end, prp_seed, bit_width, index_prefix_dir)
            return plaintext + mask
        except Exception as e:
            raise HomoEncrypException(f'Flashe Encryption Failed: {e.args[0]}')

    def decrypt(self, cyphertext:Flashev2CTxt) -> NDArray:
        try:
            unmask = _prepare_mask_multiple(self.prp, self.bit_width, cyphertext)
            return cyphertext.ciphertext - unmask
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

    x1 = np.array([1,1,1,1,1])
    x2 = np.array([2,2,2,2,2])

    cypher = Flashev2Cypher.create_with_autoseed(16)
    S1 = {
        Flashev2Cypher.concat_prefix(0,1): 1,
        Flashev2Cypher.concat_prefix(0,2): -1,
    }
    S2 = {
        Flashev2Cypher.concat_prefix(0,2): 1,
        Flashev2Cypher.concat_prefix(0,3): -1,
  
    }
    cx1 = cypher.encrypt(x1, Flashev2CypherParams(0, 5, S1))
    cx2 = cypher.encrypt(x2, Flashev2CypherParams(0 ,5, S2))
    cx3 = cx1 + cx2*2
    x3 = cypher.decrypt(cx3)
    print(x3)
    print(cx3)
    print(cx3.to_bytes())
    print(Flashev2CTxt.from_bytes(cx3.to_bytes()))