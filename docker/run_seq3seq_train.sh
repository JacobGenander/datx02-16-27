#!/bin/bash
source_dir=$(pwd)/code
data_dir=~/data
output_dir=$(pwd)/seq3seq

echo "Building docker image"
docker build -t genhead/seq3seq:gpu -f ./docker/Dockerfile . 

echo "Start roughly one hour training"
echo $source_dir
./docker/docker_run_gpu.sh --rm -t -w="/" -v $output_dir:/seq3seq -v $data_dir:/data -v $source_dir:/code genhead/seq3seq:gpu -gpu /code/seq3seq/train.sh
