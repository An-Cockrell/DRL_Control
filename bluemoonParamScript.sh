#!/bin/bash
#SBATCH --partition=week
#SBATCH --ntasks=400
#SBATCH --mem-per-cpu=200M
#SBATCH --time=120:00:00
#SBATCH --job-name=ParamSweep --output=%x.out
# SBATCH --mail-user=$daleblarie@gmail.com
# SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}
# Executable section: echoing some Slurm data
echo "Starting sbatch script myscript.sh at:`date`"
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"


source ~/scratch/mypy/bin/activate

mpiexec -n 400 python3 param_sweep_iirabm.py
