#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=10:00:00
#SBATCH --output ./sout/%A


unset LD_PRELOAD
source /etc/profile.d/modules.sh

module purge

module load exfel exfel-python openmpi-no-python


mpirun -n 8 -- python calc_run_var.py $@

