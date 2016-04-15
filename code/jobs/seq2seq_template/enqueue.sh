#!/bin/bash

#create log dir
[ -d log ] || mkdir log 

#submit job
qsub -cwd \
  -e ./log/error \
  -o ./log/output \
  ./$1 

