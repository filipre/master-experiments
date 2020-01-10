#!/bin/bash
DELAY_METHOD="constant" MAX_DELAY="1" LR="0.1" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="10" LR="0.1" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="100" LR="0.1" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="1000" LR="0.1" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="10000" LR="0.1" sbatch sgd.sbatch

DELAY_METHOD="constant" MAX_DELAY="1" LR="0.01" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="10" LR="0.01" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="100" LR="0.01" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="1000" LR="0.01" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="10000" LR="0.01" sbatch sgd.sbatch

DELAY_METHOD="constant" MAX_DELAY="1" LR="0.001" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="10" LR="0.001" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="100" LR="0.001" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="1000" LR="0.001" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="10000" LR="0.001" sbatch sgd.sbatch

DELAY_METHOD="constant" MAX_DELAY="1" LR="0.0001" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="10" LR="0.0001" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="100" LR="0.0001" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="1000" LR="0.0001" sbatch sgd.sbatch
DELAY_METHOD="constant" MAX_DELAY="10000" LR="0.0001" sbatch sgd.sbatch
