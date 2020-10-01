#!/bin/bash
#SBATCH --partition=ib --constraint=haswell_1
#SBATCH --ntasks=160
#SBATCH --time=24:00:00
#SBATCH --job-name=ParamSweep --output=%x.out
echo "we got to the job name"
module purge
module load mpi/openmpi-3.1.6-slurm-ib-verbs

cd ${SLURM_SUBMIT_DIR}
# Executable section: echoing some Slurm data
echo "Starting sbatch script myscript.sh at:`date`"
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"


source ~/scratch/mypy/bin/activate


mpiexec -n 160 python parameter_sweep.py
