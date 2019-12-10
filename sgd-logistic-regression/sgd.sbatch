#!/bin/bash
#SBATCH --job-name="SGD"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/filip/master-experiments/sgd-logistic-regression/logs/sgd-%j.out
#SBATCH --error=/usr/stud/filip/master-experiments/sgd-logistic-regression/logs/sgd-%j.out
source /usr/stud/filip/anaconda3/bin/activate base
srun python sgd.py --epochs=100 --delay-method=$DELAY_METHOD --max-delay=$MAX_DELAY --lr=$LR
