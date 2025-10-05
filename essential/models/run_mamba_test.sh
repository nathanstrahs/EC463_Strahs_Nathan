#!/bin/bash -l
#$ -P textconv
#$ -N run_mamba_test
#$ -pe omp 1
#$ -o mamba_test.out
#$ -e mamba_test.err
#$ -l gpus=1
#$ -l gpu_c=7.0

# Move into the directory from which qsub was run
cd /projectnb/textconv/nstrahs/EC463_Strahs_Nathan/essential/models

# Activate your virtual environment
source /projectnb/textconv/nstrahs/venvs/mamba_only/bin/activate

# Load required modules
module load cuda/11.8

# Run the test script
python test_mamba_implementation.py
