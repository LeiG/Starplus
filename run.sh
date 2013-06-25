#!/bin/bash

cd $PBS_O_WORKDIR

python main.py $PBS_JOBID
