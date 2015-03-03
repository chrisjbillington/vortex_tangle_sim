from __future__ import print_function

from mpi4py import MPI
import socket
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

x = 'hello from MPI process {0} of {1}, Running on {2}!\n'.format(rank, size-1, socket.gethostname())

with h5py.File('test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:
    dset = f.create_dataset('test', (size,), dtype=int)
    dset[rank] = rank**2

if rank:
    x = comm.recv(source=rank-1) + x
if rank < size-1:
    comm.send(x, dest=rank+1)
else:
    print(x)
