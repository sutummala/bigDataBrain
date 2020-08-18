#!/bin/bash
#SBATCH -p medium
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o outfile-%J

${NMRI_TOOLS}/conda_nipype/bin/python /usr/users/tummala/python/main_structural_grid.py $1 $2
