#!/bin/bash
#SBATCH --job-name=1080-12gpus

#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=7        # cpu-cores per task (>1 if multi-threaded tasks), (28 CPUS max on 1080ti nodes)
#SBATCH --mem=100GB              # make sure we use all the memory available        

#SBATCH --open-mode=append
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:4

#SBATCH --time=60:00:00

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
