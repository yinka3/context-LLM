import hashlib
from typing import Union, List

import numpy as np

class SharedID:
    shared_id: int = 0


def hash_password(password):
    hash_pws = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return hash_pws


def stoi_ids(ids: List[str], numpify = False):
    assert type(ids[0]) == str

    num_ids = []
    for id in ids:
        _, id_ = id.split("_")
        num_ids.append(int(id_))
    
    if numpify == True:
        num_ids = np.array(num_ids, dtype=np.int64)
    
    return num_ids



