#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts
#PBS -M ezequiel.centofanti@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N dataset_gen
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=72:00:00
# Request number of cores (n_machines:ppn=n_cores)
#PBS -l nodes=1:ppn=32

# Activate conda environment
module load tensorflow/2.7

cd /home/ecentofanti/wf-SEDs/WFE_sampling_test/scripts/

python ./train-eval-plot-script-click.py \
    --model param \
    --n_epochs_param 2 2 \
    --n_epochs_non_param 2 2 \
    --saved_model_type checkpoint \
    --saved_cycle cycle2 \
    --total_cycles 2 \
    --base_id_name _testing_auto_ \
    --suffix_id_name 2c --suffix_id_name 5c --suffix_id_name 1k --suffix_id_name 2k \
    --star_numbers 200 --star_numbers 500 --star_numbers 1000 --star_numbers 2000 \
    --plots_folder plots/testing_plot_folder/ \

# Return exit code
exit 0
