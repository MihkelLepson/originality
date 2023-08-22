import ctypes
import numpy as np
from typing import Tuple, List, Dict, Optional

def check_originality(targets,
                      references,
                      return_max = True) -> np.ndrarray:

  
    if return_max:
        lcs = np.zeros((len(targets), len(references)))
    else:
        lcs = np.zeros(len(targets))

    references_tokenized = references  

    # Check the OS type

    # Load the CUDA C++ shared library
    cuda_module = ctypes.CDLL('./.so')

    # Declare the function signature
    cuda_module.cudaLcs.restype = None
    cuda_module.cudaLcs.argtypes = [ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_int),
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int]
    
    divide_points = np.zeros(len(references_tokenized)+1, dtype=np.int32)
    size_org = targets.shape[0]
    size_gen = references.shape[0]
    size_div = divide_points.shape[0]

    return lcs
