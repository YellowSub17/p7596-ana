#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=10:00:00
#SBATCH --output ./sout/%A


unset LD_PRELOAD
source /etc/profile.d/modules.sh

module purge
module load exfel exfel-python openmpi-no-python

echo $(date)

echo "$@"

mpirun -n 32 -- python calc_run_mean.py $@

echo $(date)

