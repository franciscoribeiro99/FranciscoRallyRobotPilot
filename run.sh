#!/bin/bash
#SBATCH --job-name=Rally # Job name
#SBATCH --output=output.txt # Standard output file
#SBATCH --error=error.txt # Standard error file
#SBATCH --tasks=1 # Number of tasks
#SBATCH --gpus-per-node=1 # Require GPUs
#SBATCH --time=0-10:00 # Maximum runtime (D-HH:MM)
#SBATCH --nodelist=calypso0 # Specific nodes


echo STARTING AT `date`

#run the code and output the console print on output.txt file
python3 train.py


echo FINISHED at `date`