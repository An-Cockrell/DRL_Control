#!/bin/bash
#SBATCH --partition=dggpu
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=1
# Request GPUs
#SBATCH --gres=gpu:1
# Request memory
#SBATCH --mem-per-cpu=200G
#SBATCH --time=48:00:00
# Name job
#SBATCH --job-name=DDPGp05XL

# Name output file
#SBATCH --output=ddpp05XL.out
# Set email address (for user with email "usr1234@uvm.edu")
#SBATCH --mail-user=daleblarie@gmail.com
# Request email to be sent at begin and end, and if fails
#SBATCH --mail-type=FAIL
# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}
# Executable section: echoing some Slurm data
echo "Starting sbatch script myscript.sh at:`date`"
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"
echo "GPU(s):          ${CUDA_VISIBLE_DEVICES}"


source ~/scratch/mypy/bin/activate
PATH=/gpfs3/arch/x86_64-rhel7/cuda-10.0/bin:${PATH}
LD_LIBRARY_PATH=/gpfs3/arch/x86_64-rhel7/cuda-10.0/lib64:${LD_LIBRARY_PATH}

python3 -u DDPG.py
