from Pyfhel import Pyfhel, PyCtxt
import numpy as np
from heflp.secureproto.homoencrypschemes.cypher import CypherBase, HomoEncrypException
from typing import Optional, List, Dict, Union
import os

DEFAULT_PARAMS = {
    'scheme': 'BFV',    # can also be 'bfv'
    'n': 2**14,         # Polynomial modulus degree, the num. of slots per plaintext,
                        #  of elements to be encoded in a single ciphertext in a
                        #  2 by n/2 rectangular matrix (mind this shape for rotations!)
                        #  Typ. 2^D for D in [10, 16]
    't': 2**23+1,         # Plaintext modulus. Encrypted operations happen modulo t
                        #  Must be prime such that t-1 be divisible by 2^N.
    't_bits': 24,       # Number of bits in t. Used to generate a suitable value
                        #  for t. Overrides t if specified.
    'sec': 128,         # Security parameter. The equivalent length of AES key in bits.
                        #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
                        #  More means more security but also slower computation.
}

class BFVHelper(object):
    '''Provide the transforming functions for BFV scheme'''
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


class BFVCypher(CypherBase, BFVHelper):
    def __init__(self, 
                 private_key:str,
                 public_key:str,
                 n=DEFAULT_PARAMS['n'],
                 t=DEFAULT_PARAMS['t'],
                 t_bits=DEFAULT_PARAMS['t_bits'],
                 sec=DEFAULT_PARAMS['sec']
                 ):
        super().__init__((private_key, public_key))
        self.n = n
        self.he = Pyfhel()
        self.he.contextGen(scheme='bfv', n=n, t=t, t_bits=t_bits, sec=sec)
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
            plaintext = plaintext.astype(np.int64)
            encrypted_blks = [self.he.encryptInt(plaintext[i:i+self.n]) for i in range(0, len(plaintext), self.n)]
            return encrypted_blks
        except Exception as e:
            raise HomoEncrypException(f"BFV Exception Failed: {e.args[0]}")
    
    def encrypt_to_ndarray(self, plaintext: np.ndarray):
        return np.array(self.encrypt(plaintext))

    def decrypt(self, cyphertexts: List[PyCtxt], max_length: int):
        try:
            decrypted_blks = [self.he.decryptInt(x) for x in cyphertexts]
            return np.concatenate(decrypted_blks)[:max_length]
        except Exception as e:
            raise HomoEncrypException(f"BFV Decryption Failed: {e.args[0]}")