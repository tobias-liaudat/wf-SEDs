#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts (commented)
#PBS -M ezequiel.centofanti@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N 10k_pair_train
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=72:00:00
# Request number of cores (n_machines:ppn=n_cores)
#PBS -l nodes=n03:ppn=8

# Activate conda environment
module load tensorflow/2.7

python /home/ecentofanti/wf-SEDs/WFE_sampling_test/scripts/train_eval_plot_script_click.py \
    --pupil_diameter 128 \
    --n_bins_lda 8 \
    --n_epochs_param 15 15 \
    --n_epochs_non_param 100 50 \
    --model poly \
    --model_eval poly \
    --base_id_name _full_poly_ \
    --suffix_id_name 10000_stars \
    --id_name _full_poly_10000_stars \
    --test_dataset_file test_Euclid_res_id_008_wfeRes_128.npy \
    --train_dataset_file train_Euclid_res_10000_TrainStars_id_008_wfeRes_128.npy \
    --plots_folder plots/10000_stars/ \
    --model_folder chkp/10000_stars/ \
    --chkp_save_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/chkp/10000_stars/ \
    --star_numbers 10000  \
    --cycle_def complete \
    --n_zernikes 15 \
    --gt_n_zernikes 45 \
    --d_max_nonparam 5 \
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
    --base_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/ \
    --dataset_folder /n05data/ecentofanti/WFE_sampling_test/10kstars/ \
    --metric_base_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/metrics/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \



python /home/ecentofanti/wf-SEDs/WFE_sampling_test/scripts/train_eval_plot_script_click.py \
    --pupil_diameter 128 \
    --n_bins_lda 8 \
    --n_epochs_param 15 15 \
    --n_epochs_non_param 100 50 \
    --model poly \
    --model_eval poly \
    --base_id_name _full_poly_ \
    --suffix_id_name 8000_stars \
    --id_name _full_poly_8000_stars \
    --test_dataset_file test_Euclid_res_id_008_wfeRes_128.npy \
    --train_dataset_file train_Euclid_res_8000_TrainStars_id_008_wfeRes_128.npy \
    --plots_folder plots/8000_stars/ \
    --model_folder chkp/8000_stars/ \
    --chkp_save_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/chkp/8000_stars/ \
    --star_numbers 8000  \
    --cycle_def complete \
    --n_zernikes 15 \
    --gt_n_zernikes 45 \
    --d_max_nonparam 5 \
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
    --base_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/ \
    --dataset_folder /n05data/ecentofanti/WFE_sampling_test/10kstars/ \
    --metric_base_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/metrics/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \



 python /home/ecentofanti/wf-SEDs/WFE_sampling_test/scripts/train_eval_plot_script_click.py \
    --pupil_diameter 128 \
    --n_bins_lda 8 \
    --n_epochs_param 15 15 \
    --n_epochs_non_param 100 50 \
    --model poly \
    --model_eval poly \
    --base_id_name _full_poly_ \
    --suffix_id_name 6000_stars \
    --id_name _full_poly_6000_stars \
    --test_dataset_file test_Euclid_res_id_008_wfeRes_128.npy \
    --train_dataset_file train_Euclid_res_6000_TrainStars_id_008_wfeRes_128.npy \
    --plots_folder plots/6000_stars/ \
    --model_folder chkp/6000_stars/ \
    --chkp_save_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/chkp/6000_stars/ \
    --star_numbers 6000  \
    --cycle_def complete \
    --n_zernikes 15 \
    --gt_n_zernikes 45 \
    --d_max_nonparam 5 \
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
    --base_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/ \
    --dataset_folder /n05data/ecentofanti/WFE_sampling_test/10kstars/ \
    --metric_base_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/metrics/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \



python /home/ecentofanti/wf-SEDs/WFE_sampling_test/scripts/train_eval_plot_script_click.py \
    --pupil_diameter 128 \
    --n_bins_lda 8 \
    --n_epochs_param 15 15 \
    --n_epochs_non_param 100 50 \
    --model poly \
    --model_eval poly \
    --base_id_name _full_poly_ \
    --suffix_id_name 4000_stars \
    --id_name _full_poly_4000_stars \
    --test_dataset_file test_Euclid_res_id_008_wfeRes_128.npy \
    --train_dataset_file train_Euclid_res_4000_TrainStars_id_008_wfeRes_128.npy \
    --plots_folder plots/4000_stars/ \
    --model_folder chkp/4000_stars/ \
    --chkp_save_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/chkp/4000_stars/ \
    --star_numbers 4000  \
    --cycle_def complete \
    --n_zernikes 15 \
    --gt_n_zernikes 45 \
    --d_max_nonparam 5 \
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
    --base_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/ \
    --dataset_folder /n05data/ecentofanti/WFE_sampling_test/10kstars/ \
    --metric_base_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/metrics/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \



python /home/ecentofanti/wf-SEDs/WFE_sampling_test/scripts/train_eval_plot_script_click.py \
    --pupil_diameter 128 \
    --n_bins_lda 8 \
    --n_epochs_param 15 15 \
    --n_epochs_non_param 100 50 \
    --model poly \
    --model_eval poly \
    --base_id_name _full_poly_ \
    --suffix_id_name 2000_stars \
    --id_name _full_poly_2000_stars \
    --test_dataset_file test_Euclid_res_id_008_wfeRes_128.npy \
    --train_dataset_file train_Euclid_res_2000_TrainStars_id_008_wfeRes_128.npy \
    --plots_folder plots/2000_stars/ \
    --model_folder chkp/2000_stars/ \
    --chkp_save_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/chkp/2000_stars/ \
    --star_numbers 2000  \
    --cycle_def complete \
    --n_zernikes 15 \
    --gt_n_zernikes 45 \
    --d_max_nonparam 5 \
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
    --base_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/ \
    --dataset_folder /n05data/ecentofanti/WFE_sampling_test/10kstars/ \
    --metric_base_path /home/ecentofanti/wf-SEDs/more_stars/wf-outputs/metrics/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \


# Return exit code
exit 0
