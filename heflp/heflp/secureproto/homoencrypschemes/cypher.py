from typing import List, Optional
from numpy._typing import NDArray

class HomoEncrypException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class CypherBase(object):
    '''Base class for other ciphers'''
    def __init__(self, seed) -> None:
        self.seed = seed
        
    def encrypt(self, plaintext:NDArray, params:Optional[object]=None):
        pass

    def decrypt(self, cyphertext:NDArray, params:Optional[object]=None):
        pass

    def get_seed(self):
        return self.seed

