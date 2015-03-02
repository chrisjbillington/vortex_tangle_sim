#!/bin/sh
#$ -S /bin/sh
#$ -o out
#$ -j y
#$ -l h_vmem=1G
#$ -l h_rt=00:01:00
#$ -cwd
#$ -M chrisjbillington@gmail.com
#$ -m abe
#$ -N vortex_tangle
#$ -pe orte_adv 16
export OMP_NUM_THREADS=$NSLOTS
mpirun -np  $NSLOTS python run.py
