#!/bin/bash


project_run=$(basename "$0")

#create log dir
[ -d log ] || mkdir log 

#submit job
qsub -cwd \
  -e ./log/$project_run.error \
  -o ./log/$project_run.log \
  ./docker/$project_run

