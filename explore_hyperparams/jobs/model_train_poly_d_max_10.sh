#!/bin/bash
#SBATCH --job-name=d_10_model_poly    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=d_10_model_poly%j.out  # nom du fichier de sortie
#SBATCH --error=d_10_model_poly%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
#SBATCH --array=0-4

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.7.0

# echo des commandes lancees
set -x

opt[0]="--id_name model_poly_1k --train_dataset_file train_Euclid_res_1000_TrainStars_id_008_wfeRes_128.npy"
opt[1]="--id_name model_poly_2k --train_dataset_file train_Euclid_res_2000_TrainStars_id_008_wfeRes_128.npy"
opt[2]="--id_name model_poly_4k --train_dataset_file train_Euclid_res_4000_TrainStars_id_008_wfeRes_128.npy"
opt[3]="--id_name model_poly_6k --train_dataset_file train_Euclid_res_6000_TrainStars_id_008_wfeRes_128.npy"
opt[4]="--id_name model_poly_8k --train_dataset_file train_Euclid_res_8000_TrainStars_id_008_wfeRes_128.npy"

cd $WORK/repos/wf-SEDs/explore_hyperparams/scripts/

srun python -u ./train_eval_plot_script_click.py \
    --n_bins_lda 8 \
    --pupil_diameter 128 \
    --d_max_nonparam 10 \
    --n_epochs_param 15 15 \
    --n_epochs_non_param 100 50 \
    --model poly \
    --model_eval poly \
    --cycle_def complete \
    --n_zernikes 15 \
    --gt_n_zernikes 45 \
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
    --dataset_folder /gpfswork/rech/ynx/uds36vp/datasets/10k_stars/ \
    --test_dataset_file test_Euclid_res_id_008_wfeRes_128.npy \
    --plots_folder plots/ \
    --model_folder chkp/ \
    --chkp_save_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/explore_hyperparams/wf-outputs-poly-d-10/chkp/ \
    --base_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/explore_hyperparams/wf-outputs-poly-d-10/ \
    --metric_base_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/explore_hyperparams/wf-outputs-poly-d-10/metrics/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \
    --base_id_name model_poly_ \
    --suffix_id_name 1k --suffix_id_name 2k --suffix_id_name 4k --suffix_id_name 6k --suffix_id_name 8k \
    --star_numbers 1000 --star_numbers 2000 --star_numbers 4000 --star_numbers 6000 --star_numbers 8000 \
    ${opt[$SLURM_ARRAY_TASK_ID]} \

## --star_numbers is for the final plot's x-axis. 