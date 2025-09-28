import hashlib

class SharedID:
    shared_id: int = 0


def hash_password(password):
    hash_pws = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return hash_pws



