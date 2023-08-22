import ctypes
import numpy as np
from typing import Tuple, List, Dict, Optional
from transformers import AutoTokenizer

def check_originality(targets,
                      references,
                      return_max = True,
                      tokenizer : Optional[AutoTokenizer] = None,
                      **kwargs) -> np.ndrarray:
    
    if tokenizer != None:
        a = 0
    else:
        # Check if references is tokenized
        references_tokenized = references
    
    if return_max:
        lcs = np.zeros((len(targets), len(references)))
    else:
        lcs = np.zeros(len(targets))
    
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
