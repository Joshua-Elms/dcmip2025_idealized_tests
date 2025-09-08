#!/bin/bash

#SBATCH -J 2018-2019_partial_regression
#SBATCH -p general
#SBATCH -o output.out
#SBATCH -e log.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmelms@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --mem=200GB
#SBATCH -A r00389

source /N/slate/jmelms/projects/earth2studio-cu126/.venv/bin/activate
python /N/slate/jmelms/projects/dcmip2025_idealized_tests/initial_conditions/regression_initial_condition.py