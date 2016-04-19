#!/bin/bash

mkfifo articlerand titlerand
tee articlerand titlerand < /dev/urandom > /dev/null &
shuf --random-source=titlerand   train_ids.ids20000.title3   -o train_ids.ids20000.title3   &
shuf --random-source=articlerand train_ids.ids30000.article3 -o train_ids.ids30000.article3 &
wait
rm articlerand titlerand
