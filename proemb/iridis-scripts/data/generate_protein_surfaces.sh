#!/bin/bash
#SBATCH --job-name=protein-surfaces

#SBATCH --partition=batch

#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks), (28 CPUS max on 1080ti nodes)

#SBATCH --mem=16G
#SBATCH --time=60:00:00



# pymesh does not work with older gcc
module load gcc/11.1.0  

module load conda
source activate prose

# run python code
cd $HOME/protein-embeddings/proemb/surfaces


# msms generates the surface(mesh) of a protein
export MSMS_BIN='/home/ii1g17/.conda/envs/prose/bin/msms'

# "$@" is equivalent to "$1" "$2" "$3"...
# arguments must be passed in quotes
python -u generate_scop_surfaces.py "$@"
