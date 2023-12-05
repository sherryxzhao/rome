import os
import sys
def generate_sbatch_file(job_name, email, path, python_args, folder_name):

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    sbatch_content = f"""#!/bin/bash
#SBATCH -J {job_name} #job name
#SBATCH -N1 --gres=gpu:V100:1
#SBATCH -t 480 # Duration of the job (in minutes) max 8hrs
#SBATCH --mem-per-cpu=24G
#SBATCH -q coc-ice #Queue name (where job is submitted)
#SBATCH -oReport-%j.out #combine output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL #Mail preferences
#SBATCH --mail-user={email} #email address for notifications
cd {path}
module load anaconda3/2022.05.0.1
conda activate rome
python3 -m scripts.get_scores {python_args}
"""

    # Generate sbatch file name based on job name
    _user = job_name.split('_')[-2]
    _batch = job_name.split('_')[-1]

    file_name = f"{folder_name}/get_scores_{_user}_{_batch}.sbatch"

    # Write the sbatch content to a file
    with open(file_name, 'w') as file:
        file.write(sbatch_content)
    print(f"Sbatch file '{file_name}' has been generated.")

user = int(sys.argv[1])
for i in range(10):
    # Example usage
    generate_sbatch_file(f'zsre-tracing_{user}_{i}', 'yyan376@gatech.edu', '$HOME/scratch/sherry/rome', f'ZSREeval -1 -1 {user} {i}', 'sbatch')
