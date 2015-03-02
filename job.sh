#!/bin/sh
#$ -S /bin/sh
#$ -l h_vmem=1G
#$ -l h_rt=00:01:00
#$ -cwd
#$ -M chrisjbillington@gmail.com
#$ -m abe
#$ -N vortex_tangle

python run.py
