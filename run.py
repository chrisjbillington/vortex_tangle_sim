from __future__ import print_function

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

x = 'hello from MPI process {0} of {1}!\n'.format(rank, size-1)

if rank:
    x = comm.recv(source=rank-1) + x
if rank < size-1:
    comm.send(x, dest=rank+1)
else:
    print(x)
