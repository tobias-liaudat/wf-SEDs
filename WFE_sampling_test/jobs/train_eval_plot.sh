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
    --n_epochs_param 2 2 \
    --n_epochs_non_param 2 2 \
    --model poly \
    --model_eval poly \
    --base_id_name _full_poly_WFE_res_ \
    --suffix_id_name 256 \
    --id_name _full_poly_WFE_res_256 \
    --test_dataset_file test_Euclid_res_id_004_wfeRes_256.npy \
    --train_dataset_file train_Euclid_res_200_TrainStars_id_004_wfeRes_256.npy \
    --star_numbers 200  \
    --cycle_def complete \
    --plots_folder plots/ \
    --n_zernikes 45 \
    --gt_n_zernikes 45 \
    --d_max_nonparam 2 \
    --l_rate_non_param 0.1 0.06 \
    --l_rate_param 0.01 0.004 \
    --saved_model_type checkpoint \
    --saved_cycle cycle2 \
    --total_cycles 2 \
    --use_sample_weights True \
    --l2_param 0. \
    --interpolation_type none \
    --eval_batch_size 16 \
    --train_opt True \
    --eval_opt True \
    --plot_opt True \
    --base_path /home/ecentofanti/wf-SEDs/WFE_sampling_test/wf-outputs/ \
    --dataset_folder /n05data/ecentofanti/WFE_sampling_test/multires_dataset/ \
    --metric_base_path /home/ecentofanti/wf-SEDs/WFE_sampling_test/wf-outputs/metrics/ \
    --chkp_save_path /home/ecentofanti/wf-SEDs/WFE_sampling_test/wf-outputs/chkp/ \
    --log_folder log-files/ \
    --model_folder chkp/ \
    --optim_hist_folder optim-hist/ \

# Return exit code
exit 0
