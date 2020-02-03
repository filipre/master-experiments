#!/bin/bash
#SBATCH --job-name="ADMM"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=4G
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/filip/master-experiments/admm-logistic-regression/logs/admm-%j.out
#SBATCH --error=/usr/stud/filip/master-experiments/admm-logistic-regression/logs/admm-%j.out
source /usr/stud/filip/anaconda3/bin/activate base
srun python admm.py --max-iterations=100 --number-nodes=5 --delay-method=$DELAY_METHOD --max-delay=$MAX_DELAY --split=$SPLIT --multiplier=$MULTIPLIER --rho=$RHO --lr=$LR
