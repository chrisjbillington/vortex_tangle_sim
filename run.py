from __future__ import print_function

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print('hello from MPI process %d!'%rank)
