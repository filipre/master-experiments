#!/bin/bash
#SBATCH --job-name="Async Worker ADMM"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=4G
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/filip/master-experiments/admm-logistic-regression/logs/async-worker-%j.out
#SBATCH --error=/usr/stud/filip/master-experiments/admm-logistic-regression/logs/async-worker-%j.out
source /usr/stud/filip/anaconda3/bin/activate base
WORLD_SIZE=6 RANK=$RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=29500 srun python async-worker.py --max-iterations=100 --split=$SPLIT --multiplier=$MULTIPLIER --rho=$RHO --lr=$LR --random-sleep=$RANDOM_SLEEP --constant-sleep=$CONSTANT_SLEEP
