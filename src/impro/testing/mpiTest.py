from mpi4py import MPI

list = [1,1,2,2]

print sorted(list, reverse = True)[:2]