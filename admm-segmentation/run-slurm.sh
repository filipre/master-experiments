#!/bin/bash
set -e -x

# baselines
NODES=1 SCALE_IMG_SIZE="1" TAU="10" DELTA="0.01" ALPHA="0.1" DELAY="1" DELAY_METHOD="constant" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="10" DELTA="0.01" ALPHA="0.1" DELAY="1" DELAY_METHOD="constant" sbatch admm.sbatch
NODES=1 SCALE_IMG_SIZE="1" TAU="100" DELTA="0.01" ALPHA="0.1" DELAY="1" DELAY_METHOD="constant" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="100" DELTA="0.01" ALPHA="0.1" DELAY="1" DELAY_METHOD="constant" sbatch admm.sbatch

# simulated constant delays (low tau)
NODES=4 SCALE_IMG_SIZE="1" TAU="10" DELTA="0.01" ALPHA="0.1" DELAY="2" DELAY_METHOD="constant" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="10" DELTA="0.01" ALPHA="0.1" DELAY="3" DELAY_METHOD="constant" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="10" DELTA="0.01" ALPHA="0.1" DELAY="4" DELAY_METHOD="constant" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="10" DELTA="0.01" ALPHA="0.1" DELAY="5" DELAY_METHOD="constant" sbatch admm.sbatch

# simulated uniform delays (low tau)
NODES=4 SCALE_IMG_SIZE="1" TAU="10" DELTA="0.01" ALPHA="0.1" DELAY="2" DELAY_METHOD="uniform" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="10" DELTA="0.01" ALPHA="0.1" DELAY="3" DELAY_METHOD="uniform" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="10" DELTA="0.01" ALPHA="0.1" DELAY="4" DELAY_METHOD="uniform" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="10" DELTA="0.01" ALPHA="0.1" DELAY="5" DELAY_METHOD="uniform" sbatch admm.sbatch

# simulated constant delays (high tau)
NODES=4 SCALE_IMG_SIZE="1" TAU="100" DELTA="0.01" ALPHA="0.1" DELAY="2" DELAY_METHOD="constant" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="100" DELTA="0.01" ALPHA="0.1" DELAY="3" DELAY_METHOD="constant" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="100" DELTA="0.01" ALPHA="0.1" DELAY="4" DELAY_METHOD="constant" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="100" DELTA="0.01" ALPHA="0.1" DELAY="5" DELAY_METHOD="constant" sbatch admm.sbatch

# simulated uniform delays (high tau)
NODES=4 SCALE_IMG_SIZE="1" TAU="100" DELTA="0.01" ALPHA="0.1" DELAY="2" DELAY_METHOD="uniform" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="100" DELTA="0.01" ALPHA="0.1" DELAY="3" DELAY_METHOD="uniform" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="100" DELTA="0.01" ALPHA="0.1" DELAY="4" DELAY_METHOD="uniform" sbatch admm.sbatch
NODES=4 SCALE_IMG_SIZE="1" TAU="100" DELTA="0.01" ALPHA="0.1" DELAY="5" DELAY_METHOD="uniform" sbatch admm.sbatch
