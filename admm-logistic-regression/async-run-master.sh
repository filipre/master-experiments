#!/bin/bash
set -e -x
docker build -t master-pytorch -f async-master.Dockerfile .

docker run \
    --name master \
    --env MASTER_ADDR='127.0.0.1' \
    --env MASTER_PORT=29500 \
    --env RANK=0 \
    --env WORLD_SIZE=5 \
    -p 29500:29500 \
    -v ~/University/master/experiments/master-experiments/admm-logistic-regression/graphs:/workspace/graphs \
    --rm -it master-pytorch
