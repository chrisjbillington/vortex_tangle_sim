#!/bin/sh
#$ -S /bin/sh
#$ -o stdout
#$ -j y
#$ -l h_vmem=1G
#$ -l h_rt=00:01:00
#$ -cwd
#$ -M chrisjbillington@gmail.com
#$ -m abe
#$ -N vortex_tangle
#$ -pe smp 2
#$ -pe smp 2
export OMP_NUM_THREADS=$NSLOTS
mpiexec python run.py
