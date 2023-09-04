#!/bin/bash


#S#BATCH --begin=now+10hour
#SBATCH --job-name=rtx8000-4gpu
#SBATCH --partition=ecsstaff
#SBATCH --account=ecsstaff

#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=8   # cpu-cores per task (>1 if multi-threaded tasks) - 60 max for ecsstaff partition but in reality it is 32 (4 tasks x 8 cores)  

#SBATCH --open-mode=append
#SBATCH --gres=gpu:4
#SBATCH --time=120:00:00
#SBATCH --mem=320G



# -----rtx8000 complain in multi gpu training-----
# "The NCCL_P2P_DISABLE variable disables the peer to peer (P2P) transport,
# which uses CUDA direct access between GPUs, using NVLink or PCI."
export NCCL_P2P_DISABLE=1

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_PORT="$MASTER_PORT

# this is not set automatically for some reason
export SLURM_GPUS_ON_NODE=4
echo "SLRUM_GPUS_ON_NODE="$SLURM_GPUS_ON_NODE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module load conda
source activate prose

cd $HOME/protein-embeddings/proemb

# running python file for each individual process (GPU)
# "$@" is equivalent to "$1" "$2" "$3"...
# arguments must be passed in quotes
srun python -u train_prose_multitask.py "$@"

