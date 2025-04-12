#!/bin/bash -l

#SBATCH -J attractor_test
#SBATCH -p gpu
#SBATCH -o attractor_%j.txt
#SBATCH -e attractor_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmelms@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --time=02:00:00
#SBATCH --mem=700G
#SBATCH -A r00389

# env
conda activate /N/u/jmelms/BigRed200/envs/earth2mip

# vars
export WORLD_SIZE=1
export MODEL_REGISTRY=/N/slate/jmelms/projects/models

#Run your program
python test_bouvier_atmosphere.py