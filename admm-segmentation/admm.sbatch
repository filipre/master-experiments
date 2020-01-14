#!/bin/bash
#SBATCH --job-name="ADMM"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/filip/master-experiments/admm-segmentation/logs/admm-%j.out
#SBATCH --error=/usr/stud/filip/master-experiments/admm-segmentation/logs/admm-%j.out
source /usr/stud/filip/anaconda3/bin/activate base
srun python admm.py --max-iterations=100 --nodes=5 --scale-img-size=$SCALE_IMG_SIZE --tau=$TAU --delta=$DELTA --alpha=$ALPHA --delay=$DELAY --delay-method=$DELAY_METHOD
