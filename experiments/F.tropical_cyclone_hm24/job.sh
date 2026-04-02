#!/bin/bash

#SBATCH -J full_run_0402_small_IC
#SBATCH -p hopper
#SBATCH -q hopper
#SBATCH -o output.out
#SBATCH -e log.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmelms@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=03:00:00
#SBATCH --mem=120GB
#SBATCH -A r00389

source /N/slate/jmelms/projects/earth2studio-cu126/.venv/bin/activate
python /N/slate/jmelms/projects/dcmip2025_idealized_tests/experiments/F.tropical_cyclone_hm24/1.run_experiment.py
python /N/slate/jmelms/projects/dcmip2025_idealized_tests/experiments/F.tropical_cyclone_hm24/2.run_analysis.py
echo "Completed running and visualizing tropical cyclones"