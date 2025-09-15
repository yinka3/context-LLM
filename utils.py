import hashlib

def hash_password(password):
    hash_pws = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return hash_pws