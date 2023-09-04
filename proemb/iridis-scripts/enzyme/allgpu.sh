#!/bin/bash
#SBATCH --job-name=enzyme-all-gpu
#SBATCH --account=ecsstaff

#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks), (28 CPUS max on 1080ti nodes)
#SBATCH --mem=30GB
#SBATCH --open-mode=append
#SBATCH --partition=gtx1080,ecsstaff,gpu
#SBATCH --gres=gpu:1

#SBATCH --time=30:00:00

module load cuda/11.7 
module load gcc/11.1.0 

module load conda
source activate prose

cd $HOME/protein-embeddings/proemb

# "$@" is equivalent to "$1" "$2" "$3"...
# arguments must be passed in quotes
python -u train_enzyme.py "$@"


