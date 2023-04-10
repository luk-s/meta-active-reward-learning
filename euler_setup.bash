#!/bin/bash
# Switch to the new software stack
env2lmod

# Set the PYTHONPATH variable
export PYTHONPATH=$PWD:$PYTHONPATH

# Assumes that your job requires 8 CPUs
export OMP_NUM_THREADS=8

# Load the newest toolchain
module load gcc/8.2.0

# Load required modules
module load python/3.9.9

# Start the virtualenvironment
source .venv/bin/activate

# Store the current date and time
printf -v DATETIME '%(%Y-%m-%d_%H:%M:%S)T' -1
