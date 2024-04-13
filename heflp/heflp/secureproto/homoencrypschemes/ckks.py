from Pyfhel import Pyfhel, PyCtxt
import numpy as np
from heflp.secureproto.homoencrypschemes.cypher import CypherBase, HomoEncrypException
from typing import Optional, List, Dict, Union
import os

DEFAULT_PARAMS = {
    'scheme': 'CKKS',   # can also be 'ckks'
    'n': 2**14,         # Polynomial modulus degree. For CKKS, n/2 values can be
                        #  encoded in a single ciphertext.
                        #  Typ. 2^D for D in [10, 15]
    'scale': 2**30,     # All the encodings will use it for float->fixed point
                        #  conversion: x_fix = round(x_float * scale)
                        #  You can use this as default scale or use a different
                        #  scale on each operation (set in HE.encryptFrac)
    'qi_sizes': [60, 30, 30, 30, 60] # Number of bits of each prime in the chain.
                        # Intermediate values should be  close to log2(scale)
                        # for each operation, to have small rounding errors.
}

class CKKSHelper(object):
    '''Provide the transforming functions for CKKS scheme'''
    def __init__(self, context:Union[bytes, Dict]) -> None:
        self.he = Pyfhel()
        if isinstance(context, bytes):
            self.he.from_bytes_context(context)
        else:
            self.he.contextGen(**context)

    def from_bytes(self, cyphertext_bytes:Union[bytes, List[bytes]]):
        if isinstance(cyphertext_bytes, list):
            return [PyCtxt(pyfhel=self.he, bytestring=x) for x in cyphertext_bytes]
        else:
            return PyCtxt(pyfhel=self.he, bytestring=cyphertext_bytes)

    def to_bytes(self, cyphertext:Union[PyCtxt, List[PyCtxt]]):
        if isinstance(cyphertext, list):
            return [x.to_bytes() for x in cyphertext]
        else:
            return cyphertext.to_bytes()
        
    def get_context_bytes(self):
        return self.he.to_bytes_context()
    
    def set_context_bytes(self, context_bytes:bytes):
        self.he.from_bytes_context(context_bytes)

class CKKSCypher(CypherBase, CKKSHelper):
    def __init__(self, 
                 private_key:str,
                 public_key:str,
                 n=2**14,
                 qi_sizes=[60, 30, 30, 30, 60],
                 scale=2**30
                 ):
        super().__init__((private_key, public_key))
        self.n = n // 2
        self.he = Pyfhel()
        self.he.contextGen(scheme='ckks', n=n, scale=scale, qi_sizes=qi_sizes)
        if os.path.exists(private_key) and os.path.exists(public_key):
            self.he.load_public_key(public_key)
            self.he.load_secret_key(private_key)
        else:
            self.he.keyGen()
            self.he.save_public_key(public_key)
            self.he.save_secret_key(private_key)

    def encrypt(self, plaintext: np.ndarray) -> List[PyCtxt]:
        # value should be a 1-d np.array
        try:
            plaintext = plaintext.astype(np.float64)
            encrypted_blks = [self.he.encryptFrac(plaintext[i:i+self.n]) for i in range(0, len(plaintext), self.n)]
            return encrypted_blks
        except Exception as e:
            raise HomoEncrypException(f"CKKS Encryption Failed: {e.args[0]}")
    
    def encrypt_to_ndarray(self, plaintext: np.ndarray):
        return np.array(self.encrypt(plaintext))

    def decrypt(self, cyphertexts: List[PyCtxt], max_length: int):
        try:
            decrypted_blks = [self.he.decryptFrac(x).astype(np.float32) for x in cyphertexts]
            return np.concatenate(decrypted_blks)[:max_length]
        except Exception as e:
            raise HomoEncrypException(f"CKKS Decryption Failed: {e.args[0]}")