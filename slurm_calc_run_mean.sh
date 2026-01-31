#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=10:00:00
#SBATCH --output ./sout/%A


unset LD_PRELOAD
source /etc/profile.d/modules.sh

module purge

module load exfel exfel-python openmpi-no-python

echo "create_run_summary_h5.py"
echo "Date: $(date)"
echo "slurm job id: ${SLURM_JOB_ID}"
echo "cmd: mpirun -n 8 -- python create_run_summary_h5.py $@"

mpirun -n 8 -- python calc_run_mean.py $@

echo "Date: $(date)"
