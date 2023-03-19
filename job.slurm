#!/bin/bash -l
#SBATCH --job-name=CYRSOXS      # Job name
#SBATCH --output=cyrsoxs.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=cyrsoxs.%j.out  # Stderr (%j expands to jobId)
#SBATCH --time=04:00:00         # walltime
#SBATCH --nodes=1               # Number of nodes requested (only one node on glados)
#SBATCH --ntasks=1              # Number of tasks(processes) (tasks distributed across nodes)
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --cpus-per-task=1       # Threads per task (all cpus will be on same node)
#SBATCH --gres=gpu:turing:1     # number and type of GPUS to use (glados only has turing)
#SBATCH --gres-flags=enforce-binding
#SBATCH --partition=gpu

set -e

if [ x$SLURM_CPUS_PER_TASK == x ]; then
    export OMP_NUM_THREADS=1
else
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# load necessary modules
source ~/.typyEnv CLEAN
typyEnv --add cyrsoxs/latest

## RUN YOUR PROGRAM ##
echo "RUNNING ON GPU"${CUDA_VISIBLE_DEVICES}
srun CyRSoXS Fibril.hdf5