#!/bin/bash

#SBATCH --job-name=train-EuroSAT
#SBATCH --account=f202500002hpcvlabistulg
#SBATCH --time=04:00:00
#SBATCH --nodes=1
##SBATCH --gpus=4
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1

BASE="$HOME/project/ssp-ai-ws"

source $BASE/env/bin/activate
cd $BASE/ssp

./train-EuroSAT-BigCNN.py 0 512 &
./train-EuroSAT-BigCNN.py 1 256 &
./train-EuroSAT-BigCNN.py 2 128 &
./train-EuroSAT-BigCNN.py 3 64 &
wait

# EOF
