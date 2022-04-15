#!/bin/bash
#SBATCH --job-name="Dataset Gen"
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=ALL
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --array=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=hello-%j-%a.log

# Activate conda environment
module load anaconda
source activate $ANACONDA_DIR

# Change location
cd /home/ecentofanti/wf-SEDs/WFE_sampling_test/scripts/

# Run code
srun python ./gen-data-multires-parallel.py $SLURM_ARRAY_TASK_ID

# Return exit code
exit 0