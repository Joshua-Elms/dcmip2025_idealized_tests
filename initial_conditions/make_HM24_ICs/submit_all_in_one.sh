#!/bin/bash

#SBATCH -J download_sample_HM24_ICs_full
#SBATCH -p general
#SBATCH -o output.out
#SBATCH -e log.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmelms@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH -A r00389

source se2s
python /N/slate/jmelms/projects/dcmip2025_idealized_tests/initial_conditions/make_HM24_ICs/all_in_one.py