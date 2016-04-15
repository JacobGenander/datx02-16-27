#!/bin/bash

#create log dir
[ -d log ] || mkdir log 

#submit job
qsub -cwd -verify \
  -e ./log/error \
  -o ./log/output \
  ./$1 

