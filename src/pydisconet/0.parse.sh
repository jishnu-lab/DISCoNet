#!/bin/bash
#SBATCH --job-name=parsing_inputs
#SBATCH -t 0-06:00:00
#SBATCH --output=outs/%x_%A.out
#SBATCH --cluster=htc
#SBATCH --mem-per-cpu=10G
##SBATCH --mail-user=swk25@pitt.edu
##SBATCH --mail-type=END,FAIL

module purge
module load python/ondemand-jupyter-python3.10
source activate /ix/djishnu/Swapnil/.conda/envs/coauth_env/
python 0.1.py -i $1 -st $2

crc-job-stats