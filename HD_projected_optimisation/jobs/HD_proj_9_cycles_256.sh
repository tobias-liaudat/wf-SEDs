#!/bin/bash
#SBATCH --job-name=9_cycles_256   # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=9_cycles_256%j.out  # nom du fichier de sortie
#SBATCH --error=9_cycles_256%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
#SBATCH --array=0-1

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.7.0

# echo des commandes lancees
set -x

# n_bins ---> number of points per SED (n_bins + 1)
opt[0]="--id_name _9_cycles_256_proj_d2_45z --project_dd_features True"
opt[1]="--id_name _9_cycles_256_no_proj_d2_45z --project_dd_features False"


cd $WORK/repos/wf-SEDs/HD_projected_optimisation/scripts/

srun python -u ./train_project_click_multi_cycle.py \
    --d_max 2 \
    --save_all_cycles True \
    --n_bins_lda 8 \
    --plot_opt True \
    --opt_stars_rel_pix_rmse True \
    --pupil_diameter 256 \
    --n_epochs_param_multi_cycle "4 4 4 4 4 4 4 4 10" \
    --n_epochs_non_param_multi_cycle "13 13 13 13 13 13 13 13 50" \
    --l_rate_non_param_multi_cycle "0.1 0.095 0.09 0.085 0.08 0.075 0.07 0.065 0.06" \
    --l_rate_param_multi_cycle "0.01 0.00925 0.0085 0.00775 0.007 0.00625 0.0055 0.00475 0.004" \
    --total_cycles 9 \
    --saved_cycle cycle9 \
    --model poly \
    --model_eval poly \
    --cycle_def complete \
    --n_zernikes 45 \
    --gt_n_zernikes 45 \
    --d_max_nonparam 5 \
    --saved_model_type checkpoint \
    --use_sample_weights True \
    --l2_param 0. \
    --interpolation_type none \
    --eval_batch_size 16 \
    --train_opt True \
    --eval_opt True \
    --dataset_folder /gpfswork/rech/ynx/uds36vp/datasets/interp_SEDs/ \
    --test_dataset_file test_Euclid_res_id_009_8_bins.npy \
    --train_dataset_file train_Euclid_res_2000_TrainStars_id_009_8_bins_sigma_0.npy \
    --plots_folder plots/ \
    --base_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/HD_projected_optimisation/wf-outputs/ \
    --metric_base_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/HD_projected_optimisation/wf-outputs/metrics/ \
    --chkp_save_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/HD_projected_optimisation/wf-outputs/chkp/8_bins/ \
    --model_folder chkp/8_bins/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \
    --base_id_name _9_cycles_256_ \
    --suffix_id_name proj_d2_45z --suffix_id_name no_proj_d2_45z \
    --star_numbers 2 --star_numbers 3 \
    ${opt[$SLURM_ARRAY_TASK_ID]} \

## --star_numbers is for the final plot's x-axis. 