#! /bin/bash

if [ -e /tmp/tensorFlow_logs ]; then
    rm /tmp/tensorFlow_logs/*
fi

echo "---------- Running simpleLSTM.py ----------"
./simpleLSTM.py

echo "---------- Launching TensorBoard -----------"
tensorboard --logdir=/tmp/tensorFlow_logs/ 2> /dev/null
