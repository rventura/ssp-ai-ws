#!/bin/bash

#SBATCH --job-name=test-EuroSAT
#SBATCH --account=f202500002hpcvlabistulg
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

BASE="$HOME/project/ssp-ai-ws"

source $BASE/env/bin/activate
cd $BASE/ssp

for n in `seq 0 3` ;
do ./test-EuroSAT-BigCNN.py model-$n.pth ;
done

# EOF
