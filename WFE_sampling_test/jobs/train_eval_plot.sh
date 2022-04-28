#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts (commented)
# PBS -M ezequiel.centofanti@cea.fr
# PBS -m ea
# Set a name for the job
#PBS -N wf-psf_train
# Join output and errors in one file
# PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=72:00:00
# Request number of cores (n_machines:ppn=n_cores)
#PBS -l nodes=n03:ppn=2

# Activate conda environment
module load tensorflow/2.7

python /home/ecentofanti/wf-SEDs/WFE_sampling_test/scripts/train_eval_plot_script_click.py \
    --model poly \
    --n_epochs_param 2 2 \
    --n_epochs_non_param 2 2 \
    --saved_model_type checkpoint \
    --saved_cycle cycle2 \
    --total_cycles 2 \
    --base_id_name _testing_auto_ \
    --suffix_id_name 2k \
    --star_numbers 2000 \
    --plots_folder plots/ \

# Return exit code
exit 0
