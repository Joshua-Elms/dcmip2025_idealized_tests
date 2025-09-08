#!/bin/bash

#SBATCH -J test_FCN3
#SBATCH -p hopper
#SBATCH -o output.out
#SBATCH -e log.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmelms@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH -A r00389

source /N/slate/jmelms/projects/earth2studio-cu126/.venv/bin/activate
python /N/slate/jmelms/projects/dcmip2025_idealized_tests/experiments/D.extratropical_cyclone_hm24/1.run_experiment.py