#!/bin/bash
#SBATCH --job-name="wfe-study-datagen"
#SBATCH --mail-user=tobias.liaudat@cea.fr
#SBATCH --mail-type=ALL
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --array=1
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=wf-study-%j-%a.log

# Activate conda environment
module load anaconda
source activate $ANACONDA_DIR

# echo des commandes lancees
set -x

# Change location
cd /feynman/work/dap/lcs/tliaudat/repo/wf-SEDs/WFE_sampling_test/scripts/

# Run code
srun python 3x-gen-data-multires-parallel_new.py $SLURM_ARRAY_TASK_ID


# Return exit code
exit 0


