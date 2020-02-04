#!/bin/bash
#SBATCH --job-name="Async Master ADMM"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/filip/master-experiments/admm-segmentation/logs/async-master-%j.out
#SBATCH --error=/usr/stud/filip/master-experiments/admm-segmentation/logs/async-master-%j.out
source /usr/stud/filip/anaconda3/bin/activate base
WORLD_SIZE=$WORLD_SIZE RANK=0 MASTER_ADDR="127.0.0.1" MASTER_PORT=29500 srun python async-master.py --max-iterations=3 --scale-img-size=$SCALE_IMG_SIZE --tau=$TAU --delta=$DELTA --alpha=$ALPHA --barrier=$BARRIER --experiment=$EXPERIMENT
