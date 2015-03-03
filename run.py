from __future__ import print_function

from mpi4py import MPI
# import socket
import h5py

import numpy as np
import time

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()

ROOT = not RANK
FINAL = (RANK == SIZE - 1)

# x = 'hello from MPI process {0} of {1}, Running on {2}!\n'.format(rank, size-1, socket.gethostname())

# with h5py.File('test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:
#     dset = f.create_dataset('test', (size,), dtype=int)
#     dset[rank] = rank**2

N = 100000
x = np.zeros((1,), dtype=int)
if ROOT:
    start_time = time.time()
for i in range(N):
    if not ROOT:
        comm.Recv(x, source=RANK-1)
    if not FINAL:
        comm.Send(x, dest=RANK+1)
    if not FINAL:
        comm.Recv(x, source=RANK+1)
    if not ROOT:
        comm.Send(x, dest=RANK-1)
if ROOT:
    time_taken = time.time() - start_time
    with h5py.File('test.hdf5', 'w') as f:
        dset = f.create_dataset('test', data=x)
        dset.attrs['n_cores'] = SIZE
        dset.attrs['time taken'] = time_taken
        dset.attrs['time_per_send'] = time_taken/(2*SIZE*N)
