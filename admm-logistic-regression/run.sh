#!/bin/bash

set -e -x

# python admm.py --max-iterations=10 --delay-method='none' --split='split' --multiplier='with'
# python admm.py --max-iterations=10 --delay-method='none' --split='split' --multiplier='without'
# python admm.py --max-iterations=10 --delay-method='none' --split='same' --multiplier='with'
# python admm.py --max-iterations=10 --delay-method='none' --split='same' --multiplier='without'

python admm.py --max-iterations=10 --delay-method='constant' --max-delay=3 --split='split' --multiplier='with'
python admm.py --max-iterations=10 --delay-method='constant' --max-delay=3 --split='split' --multiplier='without'
python admm.py --max-iterations=10 --delay-method='constant' --max-delay=3 --split='same' --multiplier='with'
python admm.py --max-iterations=10 --delay-method='constant' --max-delay=3 --split='same' --multiplier='without'


python admm.py --max-iterations=100 --delay-method='constant' --max-delay=3 --split='same' --multiplier='without' --number-nodes=3
