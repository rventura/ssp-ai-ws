#!/bin/bash

#SBATCH --job-name=train-EuroSAT
#SBATCH --account=f202500002hpcvlabistulg
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4

BASE="$HOME/project/ssp-ai-ws"

source $BASE/env/bin/activate
cd $BASE/ssp

./train_EuroSAT.py

# EOF
