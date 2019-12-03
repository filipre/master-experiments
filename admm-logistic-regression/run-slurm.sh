#!/bin/bash
set -e -x

# no delays
# no delay, no data splitting, no lag. multipliers
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="no" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="no" MULTIPLIER="no" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# no delay, no data splitting, with lag. multipliers
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="no" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="no" MULTIPLIER="yes" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# no delay, with data splitting, no lag. multipliers
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="no" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# no delay, with data splitting, with lag. multipliers
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="yes" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="none" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch

# with delays (constant, 5)
# no delay, no data splitting, no lag. multipliers
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="5" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# # no delay, no data splitting, with lag. multipliers
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="5" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# # no delay, with data splitting, no lag. multipliers
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="5" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# # no delay, with data splitting, with lag. multipliers
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="5" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
