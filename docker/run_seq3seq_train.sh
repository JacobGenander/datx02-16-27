#!/bin/bash
source_dir=$(pwd)/code
data_dir=~/data
output_dir=$(pwd)/seq3seq

echo $source_dir
./docker/docker_run_gpu.sh --rm -t -w="/" -v $output_dir:/seq3seq -v $data_dir:/data -v $source_dir:/code b.gcr.io/tensorflow/tensorflow:0.7.1-gpu /code/seq3seq/train.sh
