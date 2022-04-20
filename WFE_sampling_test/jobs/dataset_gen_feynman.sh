#!/bin/bash
#SBATCH --job-name="Dataset Gen"
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=ALL
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --array=1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --output=output_data_gen-%j-%a.log

# Activate conda environment
module load anaconda
source activate $ANACONDA_DIR
conda activate tf-gpu

# Change location
cd /feynman/work/dap/lcs/ec270266/wf-SEDs/WFE_sampling_test/scripts/

# Run code
srun python ./gen-data-multires-parallel.py $SLURM_ARRAY_TASK_ID

# Return exit code
exit 0
