#!/bin/bash
#SBATCH -J experiment #job name
#SBATCH -N1 --gres=gpu:V100:1
#SBATCH -t  240 # Duration of the job (in minites) max 8hrs
#SBATCH --mem-per-cpu=24G
#SBATCH -q coc-ice #Queue name (where job is submitted)
#SBATCH -oReport-%j.out #combine output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL #Mail preferences
#SBATCH --mail-user=yyan376@gatech.edu #email address for notifications
pwd
cd $HOME/scratch/rome
pwd
module load anaconda3/2022.05.0.1
conda activate rome
conda env list
python3 -m experiments.evaluate --alg_name=ROME --model_name=gpt2-xl --hparams_fname=gpt2-xl.json