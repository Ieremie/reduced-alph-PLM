#!/bin/bash

#SBATCH --job-name=jupyter

#SBATCH --nodes=1                # node count
#SBATCH --partition=ecsall
#SBATCH --account=ecsstaff
#SBATCH --gres=gpu:1

#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1     # total number of tasks per node
#SBATCH --cpus-per-task=8   # cpu-cores per task (>1 if multi-threaded tasks) - 60 max for ecsstaff partition but in reality it is 32 (4 tasks x 8 cores)  

#SBATCH --open-mode=append
#SBATCH --time=120:00:00
#SBATCH --mem=120G

module load cuda/11.7 
module load gcc/11.1.0 

module load conda
source activate prose

cd $HOME/protein-embeddings


#jupyter lab  --no-browser --port=8080
jupyter lab --no-browser --ip "*"
