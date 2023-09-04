#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=jupyter

#SBATCH --partition=batch

#SBATCH --time=60:00:00
#SBATCH --mem=50GB              # make sure we use all the memory available        

#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=16

echo "caca1"

module load conda
echo "caca2"
source activate prose
echo "caca3"

cd $HOME/protein-embeddings/proemb

#jupyter lab  --no-browser --port=8080
jupyter lab --no-browser --ip "*"
