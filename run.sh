#!/bin/bash

# We are using a special variable that is set by the cluster when a job runs.
mkdir $PBS_JOBID

# Change to that new directory
cd $PBS_JOBID

cat python starplus.py >>run.sh