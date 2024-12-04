#!/bin/bash
#SBATCH --job-name=training # Job name
#SBATCH --output=output.txt # Standard output file
#SBATCH --error=error.txt # Standard error file
#SBATCH --tasks=4 # Number of tasks
#SBATCH --gpus-per-node=1 # Require GPUs
#SBATCH --time=0-10:00 # Maximum runtime (D-HH:MM)
#SBATCH --nodelist=calypso0 # Specific nodes


echo STARTING AT `date`

#run the code and output the console print on output.txt file
python3 training3.py

echo FINISHED at `date`
