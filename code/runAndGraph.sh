#! /bin/bash

if [ -e /tmp/tensorFlow_logs ]; then
    rm /tmp/tensorFlow_logs/*
fi

echo "---------- Running LSTM.py ----------"
./LSTM.py

echo "---------- Launching TensorBoard -----------"
tensorboard --logdir=/tmp/tensorFlow_logs/ 2> /dev/null
