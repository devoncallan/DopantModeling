#!/bin/bash -l
#SBATCH --job-name=CyRSoXS_F4TCNQ_Perp_Matrix	# Job name
#SBATCH --output=cyrsoxs.%j.out 		# Stdout (%j expands to jobId)
#SBATCH --error=cyrsoxs.%j.out  		# Stderr (%j expands to jobId)
#SBATCH --time=168:00:00         		# walltime
#SBATCH --nodes=1               		# Number of nodes requested
#SBATCH --ntasks=1              		# Number of tasks (processes) (tasks distributed across nodes)
#SBATCH --ntasks-per-node=1     		# Tasks per node
#SBATCH --cpus-per-task=1       		# Threads per task (all cpus will be on same node)
#SBATCH --gres=gpu:1     			# number and type of GPUS to use
#SBATCH --gres-flags=enforce-binding
#SBATCH --partition=gpu
#SBATCH --mail-user=php@ucsb.edu		# Mail to you (Optional)
#SBATCH --mail-type ALL 			# Mail send you when the job starts and end (Optional)

set -e

if [ x$SLURM_CPUS_PER_TASK == x ]; then
    export OMP_NUM_THREADS=1
else
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# Activate Anaconda work environment for CyRSOXS
source /home/${USER}/.bashrc
source activate nrss 

## RUN YOUR PROGRAM ##
echo "RUNNING ON GPU"${CUDA_VISIBLE_DEVICES}
python cyrsoxs_doped.py