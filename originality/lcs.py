import ctypes
import numpy as np
from typing import Tuple, List, Dict, Optional
import os
import itertools

def check_originality(targets,
                      references,
                      return_max: Optional[bool] = True) -> np.ndarray:
    
    if return_max:
        lcs = np.zeros(len(targets))
    else:
        lcs = np.zeros((len(targets), len(references)))
    #lcs = np.empty(len(references), dtype=np.int32)
    
    # Check the OS type
    cuda_module_path = os.path.join(os.path.dirname(__file__), '..', 'cuda_code', 'cuda_lcs_module.so')  

    # Load the CUDA C++ shared library
    cuda_module = ctypes.CDLL(cuda_module_path)

    # Declare the function signature
    cuda_module.cudaLcs.restype = None
    cuda_module.cudaLcs.argtypes = [ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_int),
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int]
    
    references_array = np.array(list(itertools.chain.from_iterable(references)), dtype=np.int32)
    print("This is references_array")
    print(references_array)
    divide_points = np.zeros(len(references)+1, dtype=np.int32)

    sum = 0
    for i, reference in enumerate(references):
        sum += len(reference)
        divide_points[i+1] = sum
    print("These are divide_points")
    print(divide_points)
    size_org = targets.shape[0]
    size_gen = references_array.shape[0]
    size_div = divide_points.shape[0]

    # Call the CUDA C++ function
    cuda_module.cudaLcs(targets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        references_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        lcs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        divide_points.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        ctypes.c_int(size_org),
                        ctypes.c_int(size_gen),
                        ctypes.c_int(size_div))

    return lcs
