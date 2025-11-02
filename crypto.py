from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import hashlib

class BitcoinStyleCrypto:
    
    def __init__(self):
        self.backend = default_backend()
        self.curve = ec.SECP256K1()
    
    def generate_keypair(self):
        private_key = ec.generate_private_key(self.curve, self.backend)
        public_key = private_key.public_key()
        return private_key, public_key
    
    def save_private_key(self, private_key, filepath, password=None):
        encryption = serialization.BestAvailableEncryption(password.encode()) if password else serialization.NoEncryption()
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption
        )
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(pem)
    
    def load_private_key(self, filepath, password=None):
        with open(filepath, 'rb') as f:
            pem = f.read()
        pwd = password.encode() if password else None
        return serialization.load_pem_private_key(pem, password=pwd, backend=self.backend)
    
    def save_public_key(self, public_key, filepath):
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(pem)
    
    def load_public_key(self, filepath):
        with open(filepath, 'rb') as f:
            pem = f.read()
        return serialization.load_pem_public_key(pem, backend=self.backend)
    
    def get_key_fingerprint(self, public_key):
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        hash1 = hashlib.sha256(pem).digest()
        hash2 = hashlib.sha256(hash1).digest()
        return hash2[:20].hex()
    
    def encrypt_file(self, data: bytes, recipient_public_key) -> bytes:
        ephemeral_private = ec.generate_private_key(self.curve, self.backend)
        ephemeral_public = ephemeral_private.public_key()
        
        shared_secret = ephemeral_private.exchange(ec.ECDH(), recipient_public_key)
        
        aes_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'encryption',
            backend=self.backend
        ).derive(shared_secret)
        
        nonce = os.urandom(12)
        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(nonce), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        ephemeral_public_bytes = ephemeral_public.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
        
        result = len(ephemeral_public_bytes).to_bytes(2, 'big')
        result += ephemeral_public_bytes
        result += nonce
        result += encryptor.tag
        result += ciphertext
        
        return result
    
    def decrypt_file(self, encrypted_data: bytes, private_key) -> bytes:
        offset = 0
        
        ephemeral_pub_len = int.from_bytes(encrypted_data[offset:offset+2], 'big')
        offset += 2
        ephemeral_pub_bytes = encrypted_data[offset:offset+ephemeral_pub_len]
        offset += ephemeral_pub_len
        
        ephemeral_public = ec.EllipticCurvePublicKey.from_encoded_point(self.curve, ephemeral_pub_bytes)
        
        shared_secret = private_key.exchange(ec.ECDH(), ephemeral_public)
        
        aes_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'encryption',
            backend=self.backend
        ).derive(shared_secret)
        
        nonce = encrypted_data[offset:offset+12]
        offset += 12
        tag = encrypted_data[offset:offset+16]
        offset += 16
        ciphertext = encrypted_data[offset:]
        
        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(nonce, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    def sign_data(self, data: bytes, private_key) -> bytes:
        signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
        return signature
    
    def verify_signature(self, data: bytes, signature: bytes, public_key) -> bool:
        try:
            public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False