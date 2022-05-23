#!/bin/bash
#SBATCH --job-name="Dataset Gen 3x"
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=ALL
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --array=1
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=dataset_gen_3x-%j-%a.log

# Activate conda environment
module load anaconda
source activate $ANACONDA_DIR
conda activate tf-gpu

# Change location
cd /feynman/work/dap/lcs/ec270266/wf-SEDs/WFE_sampling_test/scripts/

# Run code
srun python ./3x-gen-data-multires-parallel.py $SLURM_ARRAY_TASK_ID

# Return exit code
exit 0
