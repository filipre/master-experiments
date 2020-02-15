#!/bin/bash
set -e -x

# baseline
DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="no" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="no" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch

# constant
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="50" LR="0.001" sbatch admm.sbatch

DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="50" LR="0.001" sbatch admm.sbatch

DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="50" LR="0.001" sbatch admm.sbatch

DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="constant" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="50" LR="0.001" sbatch admm.sbatch

# uniform
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="50" LR="0.001" sbatch admm.sbatch

DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="50" LR="0.001" sbatch admm.sbatch

DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="50" LR="0.001" sbatch admm.sbatch

DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="5" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="50" LR="0.001" sbatch admm.sbatch


# # no delays
# # no delay, no data splitting, no lag. multipliers
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="no" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="no" MULTIPLIER="no" RHO="5" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# # no delay, no data splitting, with lag. multipliers
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="no" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="no" MULTIPLIER="yes" RHO="5" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# # no delay, with data splitting, no lag. multipliers
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="no" RHO="1" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="no" RHO="5" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# # no delay, with data splitting, with lag. multipliers
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="yes" RHO="1" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="yes" RHO="5" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="constant" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
#
# # with delays (constant, 5)
# # no delay, no data splitting, no lag. multipliers
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


# DELAY_METHOD="uniform" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="2" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="3" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="4" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="6" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="7" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="8" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="9" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="10" SPLIT="yes" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
#
# DELAY_METHOD="uniform" MAX_DELAY="1" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="2" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="3" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="4" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="6" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="7" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="8" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="9" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="10" SPLIT="no" MULTIPLIER="yes" RHO="10" LR="0.001" sbatch admm.sbatch
#
# DELAY_METHOD="uniform" MAX_DELAY="1" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="2" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="3" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="4" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="6" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="7" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="8" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="9" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="10" SPLIT="yes" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
#
# DELAY_METHOD="uniform" MAX_DELAY="1" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="2" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="3" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="4" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="5" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="6" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="7" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="8" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="9" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
# DELAY_METHOD="uniform" MAX_DELAY="10" SPLIT="no" MULTIPLIER="no" RHO="10" LR="0.001" sbatch admm.sbatch
