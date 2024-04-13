from phe.util import getprimeover
from phe import paillier, PaillierPublicKey, PaillierPrivateKey
from heflp.secureproto.homoencrypschemes.cypher import CypherBase
import numpy as np

def generate_paillier_keypair_npq(n_length=paillier.DEFAULT_KEYSIZE):
    """Return a new :class:`PaillierPublicKey` and :class:`PaillierPrivateKey`.

    Add the private key to *private_keyring* if given.

    Args:
      private_keyring (PaillierPrivateKeyring): a
        :class:`PaillierPrivateKeyring` on which to store the private
        key.
      n_length: key size in bits.

    Returns:
      tuple: The generated :class:`PaillierPublicKey` and
      :class:`PaillierPrivateKey`
    """
    p = q = n = None
    n_len = 0
    while n_len != n_length:
        p = getprimeover(n_length // 2)
        q = p
        while q == p:
            q = getprimeover(n_length // 2)
        n = p * q
        n_len = n.bit_length()

    return n, p, q

class PaillierCypher(CypherBase):
    def __init__(self, n, p, q) -> None:
        super().__init__((n, p, q))
        self.publickey = PaillierPublicKey(n)
        self.privatekey = PaillierPrivateKey(self.publickey, p, q)

    @classmethod
    def create_with_autoseed(cls, n_length=paillier.DEFAULT_KEYSIZE):
        n, p, q = generate_paillier_keypair_npq(n_length)
        return PaillierCypher(n, p, q)

    def encrypt(self, plaintext: np.ndarray):
        ret = [self.publickey.encrypt(x) for x in plaintext]
        return np.array(ret)

    def decrypt(self, cyphertext: np.ndarray):
        ret = [self.privatekey.decrypt(x) for x in cyphertext]
        return np.array(ret)


if __name__ == "__main__":
    print("test")