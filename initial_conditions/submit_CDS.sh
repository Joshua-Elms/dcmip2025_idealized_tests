#!/bin/bash

#SBATCH -J 1979-2019_CDS_download
#SBATCH -p general
#SBATCH -o output.out
#SBATCH -e log.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmelms@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH -A r00389

source /N/slate/jmelms/projects/earth2studio-cu126/.venv/bin/activate
python /N/slate/jmelms/projects/dcmip2025_idealized_tests/initial_conditions/make_HM24_ICs/download_and_compute_ICs.py