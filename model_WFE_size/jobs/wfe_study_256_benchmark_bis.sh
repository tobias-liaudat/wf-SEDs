#!/bin/bash
#SBATCH --job-name=wfe_study_256_benchmark_bis_    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=wfe_study_256_benchmark_bis_%j.out  # nom du fichier de sortie
#SBATCH --error=wfe_study_256_benchmark_bis_%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
#SBATCH --array=0-4

# nettoyage des modules charges en interctif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.7.0

# echo des commandes lancees
set -x

opt[0]="--id_name _wfe_study_256_benchmark_bis_1"
opt[1]="--id_name _wfe_study_256_benchmark_bis_2"
opt[2]="--id_name _wfe_study_256_benchmark_bis_3"
opt[3]="--id_name _wfe_study_256_benchmark_bis_4"
opt[4]="--id_name _wfe_study_256_benchmark_bis_5"

cd $WORK/repo/wf-psf/long-runs/

srun python -u ./train_eval_plot_script_click.py \
    --model poly \
    --model_eval poly \
    --pupil_diameter 256 \
    --cycle_def complete \
    --plots_folder plots/wfe_study_bis/ \
    --base_id_name _wfe_study_256_benchmark_bis_ \
    --test_dataset_file test_Euclid_res_id_004_wfeRes_256.npy \
    --train_dataset_file train_Euclid_res_2000_TrainStars_id_004_wfeRes_256.npy \
    --n_bins_lda 20 \
    --n_epochs_non_param 100 50 \
    --n_epochs_param 15 15 \
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
    --base_path /gpfswork/rech/ynx/ulx23va/repo/wf-SEDs/model_WFE_size/wf-outputs/ \
    --dataset_folder /gpfswork/rech/ynx/ulx23va/repo/wf-SEDs/WFE_sampling_test/multires_dataset/4096/ \
    --metric_base_path /gpfswork/rech/ynx/ulx23va/repo/wf-SEDs/model_WFE_size/wf-outputs/metrics/wfe_study_bis/ \
    --chkp_save_path /gpfswork/rech/ynx/ulx23va/repo/wf-SEDs/model_WFE_size/wf-outputs/chkp/wfe_study_bis/ \
    --log_folder log-files/wfe_study_bis/ \
    --model_folder chkp/wfe_study_bis/ \
    --optim_hist_folder optim-hist/wfe_study_bis/ \
    --suffix_id_name 1 --suffix_id_name 2 --suffix_id_name 3 --suffix_id_name 4 --suffix_id_name 5 \
    --star_numbers 1 --star_numbers 2 --star_numbers 3 --star_numbers 4 --star_numbers 5 \
    ${opt[$SLURM_ARRAY_TASK_ID]} \
