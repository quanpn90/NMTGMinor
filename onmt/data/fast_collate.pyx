# cython: language_level=3

import numpy as np
cimport cython
cimport numpy as np

DTYPE=np.int64
ctypedef np.int64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=2] fast_collate(list data, int align_right, long data_size):

    # first we need to find the
    cdef np.ndarray[DTYPE_t, ndim=1]

    cdef long seq_len = 0
    for i in range(data_size):
        if seq_len < data[i].shape[0]:
            seq_len = data[i].shape[0]


    cdef np.ndarray[DTYPE_t, ndim=1] sample
    cdef i

    cdef np.ndarray output = np.zeros([batch_size, seq_len], dtype=DTYPE)

    for i in range(data_size):


    return 0