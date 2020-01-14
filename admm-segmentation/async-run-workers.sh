#!/bin/bash
set -e -x
docker build -t worker-pytorch -f async-worker.Dockerfile .

for i in 1 2 3
do
    echo "Start worker $i"
    docker run \
        --name "worker$i" \
        --env RANK=$i \
        --env WORLD_SIZE=5 \
        --env MASTER_ADDR=$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' master) \
        --env MASTER_PORT=29500 \
        -p "2950$i:29500" \
        --rm -d worker-pytorch
done

# explicit for 4 to see output
for i in 4
do
    echo "Start worker $i"
    docker run \
        --name "worker$i" \
        --env RANK=$i \
        --env WORLD_SIZE=5 \
        --env MASTER_ADDR=$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' master) \
        --env MASTER_PORT=29500 \
        -p "2950$i:29500" \
        --rm -it worker-pytorch
done
