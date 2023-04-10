#!/bin/bash
# Switch to the new software stack
env2lmod

# Set environment variables
echo "Make sure that you did set the PYTHONPATH variable! Use the ../../../env_setup.bash script for this."

# Assumes that your job only requres 2 CPUs
export OMP_NUM_THREADS=4

# Load the newest toolchain
module load gcc/8.2.0

# Load required modules
module load python/3.9.9

# Start the virtualenvironment
source ../../../.venv/bin/activate

# Store the current date and time
printf -v DATETIME '%(%Y-%m-%d_%H:%M:%S)T' -1

# Submit the program
# bsub -W 24:00 -n 64 -R rusage[mem=4000] -N -B -o logs/output_$DATETIME.txt python test_reward_learners_parallel.py