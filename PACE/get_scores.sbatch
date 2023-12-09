#!/bin/bash
#SBATCH -J zsre-tracing_10102_end #job name
#SBATCH -N1 --gres=gpu:V100:1
#SBATCH -t 480 # Duration of the job (in minites) max 8hrs
#SBATCH --mem-per-cpu=24G
#SBATCH -q coc-ice #Queue name (where job is submitted)
#SBATCH -oReport-%j.out #combine output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL #Mail preferences
#SBATCH --mail-user=[your email] #email address for notifications
cd [your rome folder path from $HOME]
module load anaconda3/2022.05.0.1
conda activate rome
python3 -m scripts.get_scores ZSREeval 10102
