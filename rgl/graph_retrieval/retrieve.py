from rgl.graph_retrieval import libretrieval

retrieve = libretrieval.retrieve
batch_retrieve = libretrieval.batch_retrieve

# import numpy as np
# import ctypes
# import os

# def load_c_retrieve():
#     lib_name = "libretrieval.dll" if os.name == "nt" else "libretrieval.so"
#     lib_path = os.path.join(os.path.dirname(__file__), lib_name)
#     lib = ctypes.CDLL(lib_path)
#     c_retrieve = lib.retrieve

#     lib.retrieve.argtypes = [
#         ctypes.POINTER(ctypes.c_int),  # src
#         ctypes.POINTER(ctypes.c_int),  # dst
#         ctypes.POINTER(ctypes.c_int),  # seed
#         ctypes.c_int,  # num_edge
#         ctypes.c_int,  # num_seed
#         ctypes.POINTER(ctypes.c_int),  # num_retrieved
#     ]
#     lib.retrieve.restype = ctypes.POINTER(ctypes.c_int)

#     def ndcg_score(src, dst, seeds):

#         src = list(src)
#         dst = list(dst)
#         num_edges = len(src)
#         seeds = list(seeds)
#         num_seeds = len(seeds)
#         num_retrieved = ctypes.c_int(0)
#         rel_type = ctypes.c_int * num_edges

#         result_ptr = c_retrieve(
#             rel_type(*src),
#             rel_type(*dst),
#             rel_type(*seeds),
#             ctypes.c_int(num_edges),
#             ctypes.c_int(num_seeds),
#             ctypes.byref(num_retrieved),
#         )
#         result = [result_ptr[i] for i in range(num_retrieved.value)]
#         return result

#     return ndcg_score

# retrieve = load_c_retrieve()
