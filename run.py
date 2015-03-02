from __future__ import print_function

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('hello from MPI process {0} of {1}!'.format(rank, size-1))
