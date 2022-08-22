#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts (commented)
# PBS -M tobias.liaudat@cea.fr
# PBS -m ea
# Set a name for the job
#PBS -N gen_dataset_x3
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=20:00:00
# Request number of cores (n_machines:ppn=n_cores)
#PBS -l nodes=n01:ppn=24

# Activate conda environment
module load tensorflow/2.7

cd /n05data/tliaudat/repo/wf-SEDs/WFE_sampling_test/scripts/

python ./3x-gen-data-multires-parallel_new.py

# Return exit code
exit 0


