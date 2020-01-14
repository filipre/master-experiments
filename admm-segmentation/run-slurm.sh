#!/bin/bash
set -e -x

SCALE_IMG_SIZE="0.25" TAU="10" DELTA="0.01" ALPHA="0.1" DELAY="1" DELAY_METHOD="constant" sbatch admm.sbatch
