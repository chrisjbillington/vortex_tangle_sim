from __future__ import print_function

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

x = 'hello from MPI process {0} of {1}!'.format(rank, size-1)
print(x)
with open('test.txt','a') as f:
    f.write(x + '\n')
