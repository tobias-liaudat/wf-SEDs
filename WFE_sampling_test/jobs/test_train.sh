#!/bin/sh

python3 ../scripts/train_eval_plot_script_click.py \
    --model mccd \
    --id_name -testing_auto_200_stars  \
    --n_epochs_param 2 2 \
    --n_epochs_non_param 2 2 \
    --saved_model_type checkpoint \
    --saved_cycle cycle2 \
    --total_cycles 2 \
    --base_id_name _testing_auto_ \
    --suffix_id_name 2c \
    --star_numbers 200 \
    --plots_folder plots/ \
    --base_path /Users/ec270266/Desktop/Stage-CEA/wf-SEDs/WFE_sampling_test/wf-outputs/ \
    --chkp_save_path ../wf-outputs/chkp/ \
    --dataset_folder /Users/ec270266/Desktop/Stage-CEA/output/4096/ \
    --train_dataset_file train_Euclid_res_200_TrainStars_id_004_wfeRes_256.npy \
    --test_dataset_file test_Euclid_res_id_004_wfeRes_256.npy \
    --metric_base_path ../wf-outputs/metrics/ \