#!/bin/bash
#SBATCH --job-name=interp_train    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=interp_train_%j.out  # nom du fichier de sortie
#SBATCH --error=interp_train_%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
#SBATCH --array=0-3

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.7.0

# echo des commandes lancees
set -x

opt[0]="--id_name _interp_32_bins_sigma_0 --train_dataset_file train_Euclid_res_2000_TrainStars_id_009_8_bins_sigma_0.npy"
opt[1]="--id_name _interp_32_bins_sigma_005 --train_dataset_file train_Euclid_res_2000_TrainStars_id_009_8_bins_sigma_0.005.npy"
opt[2]="--id_name _interp_32_bins_sigma_01 --train_dataset_file train_Euclid_res_2000_TrainStars_id_009_8_bins_sigma_0.01.npy"
opt[3]="--id_name _interp_32_bins_sigma_02 --train_dataset_file train_Euclid_res_2000_TrainStars_id_009_8_bins_sigma_0.02.npy"

cd $WORK/repo/wf-psf/long-runs/

srun python -u ./train_eval_plot_script_click.py \

    --n_bins = 33
    --pupil_diameter 256 \
    --n_epochs_param 15 15 \
    --n_epochs_non_param 100 50 \
    --model poly \
    --model_eval poly \
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

    --dataset_folder /gpfswork/rech/ynx/uds36vp/datasets/interp_SEDs/ \
    --test_dataset_file test_Euclid_res_id_009_32_bins.npy \

    --plots_folder plots/32_bins/ \
    --model_folder chkp/32_bins/ \
    --chkp_save_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/interp_SED_model/wf-outputs/chkp/32_bins/ \
    --base_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/interp_SED_model/wf-outputs/ \
    
    --metric_base_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/interp_SED_model/wf-outputs/metrics/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \

    --base_id_name _interp_32_bins_ \
    --suffix_id_name sigma_0 --suffix_id_name sigma_005 --suffix_id_name sigma_01 --suffix_id_name sigma_02 \
    --star_numbers 2000  \

    ${opt[$SLURM_ARRAY_TASK_ID]} \
