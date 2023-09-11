import ctypes
import numpy as np
from typing import Tuple, List, Dict, Optional
import os
import itertools

def get_divide_points(input: List[List[int]]) -> np.ndarray:
    divide_points = np.zeros(len(input)+1, dtype=np.int32)
    sum = 0
    for i, ref in enumerate(input):
        sum += len(ref)
        divide_points[i+1] = sum
    
    return divide_points

def check_originality(targets: List[List[int]],
                      references: List[List[int]],
                      return_max: Optional[bool] = False) -> np.ndarray:
    
    #if return_max:
    #    lcs = np.empty(len(targets), dtype=np.int32)
    #else:
    #    lcs = np.empty(len(targets)*len(references), dtype=np.int32)
    lcs = np.empty(len(targets)*len(references), dtype=np.int32)
    
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
                                    ctypes.POINTER(ctypes.c_int),
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_bool]
    
    references_array = np.array(list(itertools.chain.from_iterable(references)), dtype=np.int32)
    targets_array = np.array(list(itertools.chain.from_iterable(targets)), dtype=np.int32)
    
    divide_points_ref = get_divide_points(references)
    divide_points_tar = get_divide_points(targets)
    
    size_ref = references_array.shape[0]
    size_div_ref = divide_points_ref.shape[0]
    size_div_tar = divide_points_tar.shape[0]

    # Call the CUDA C++ function
    cuda_module.cudaLcs(targets_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        references_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        lcs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        divide_points_tar.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        divide_points_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        ctypes.c_int(size_ref),
                        ctypes.c_int(size_div_tar),
                        ctypes.c_int(size_div_ref),
                        False)

    lcs = lcs.reshape((len(targets),len(references)))

    return lcs
