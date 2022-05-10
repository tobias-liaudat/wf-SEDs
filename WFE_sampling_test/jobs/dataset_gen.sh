#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts
# PBS -M ezequiel.centofanti@cea.fr
# PBS -m ea
# Set a name for the job
#PBS -N dataset_gen_3x
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=72:00:00
# Request number of cores
#PBS -l nodes=n18:ppn=24

# Activate conda environment
module load tensorflow/2.7

cd /home/ecentofanti/wf-SEDs/WFE_sampling_test/scripts/

python ./3x-gen-data-multires-parallel.py

# Return exit code
exit 0
