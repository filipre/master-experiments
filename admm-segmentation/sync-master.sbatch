#!/bin/bash
#SBATCH --job-name="Sync ADMM"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/filip/master-experiments/admm-segmentation/logs/sync_admm-%j.out
#SBATCH --error=/usr/stud/filip/master-experiments/admm-segmentation/logs/sync_admm-%j.out
source /usr/stud/filip/anaconda3/bin/activate base
srun python huberROF.py
