#!/bin/bash

# Run one task on one node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=pathpave

# Make 48 cores available to our task (otherwise defaults to 1)
#SBATCH --cpus-per-task=100

# Use any of the compute nodes in the 'all' partition
#SBATCH --partition=members

# Redirect output and error output
#SBATCH --output=job.out
#SBATCH --error=job.err

#!/bin/bash
source ./venv/bin/activate

python cluster.py
