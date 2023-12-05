import os
import sys
def generate_sbatch_content(job_name, email, path, start, end, data_file_path, run_dir_name, is_edit_max):
    sbatch_content = \
f"""#!/bin/bash
#SBATCH -J {job_name} #job name 
#SBATCH -N1 --gres=gpu:1 -C V100-32GB
#SBATCH -t 480 # Duration of the job (in minutes) max 8hrs
#SBATCH --mem-per-cpu=24G
#SBATCH -q coc-ice #Queue name (where job is submitted)
#SBATCH -oReport-%j.out #combine output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL #Mail preferences
#SBATCH --mail-user={email} #email address for notifications
cd {path}
module load anaconda3/2022.05.0.1
conda activate rome
python3 -m experiments.evaluate --alg_name=ROME --model_name=gpt2-xl --hparams_fname=gpt2-xl.json --ds_name=zsre --data_file_name={data_file_path} --start={start} --end={end} --continue_from_run {run_dir_name}"""
    sbatch_content += f' --is_edit_max' if is_edit_max else ''
    return sbatch_content

def create_tasks_for_user(user_id):

    ###### Edit Here #######
    email = 'YOUR EMAIL'
    data_path = 'data/zsre_comb_max_layer_requests.json' # put zsre_comb_max_layer_requests.json in rome/data
    path_to_rome = '../scratch/rome' # Edit if your path is differnt
    sbatch_tasks_folder = 'eval_sbatch_tasks'
    ###### END ############
    
    # Create the folder if it does not exist
    if not os.path.exists(sbatch_tasks_folder):
        os.makedirs(sbatch_tasks_folder)
    
    run_dir_name = f'run_eval_user_{user_id}'

    mapping = {1: (0, 6000), 2: (6000, 12000), 3: (12000, 18887)}
    idx = 0

    for left in range(mapping[user_id][0], mapping[user_id][1], 1000):
        right = min(left + 1000, mapping[user_id][1])
        job_name_default = f'eval_{user_id}_default_edit_p{idx}'
        job_name_max_edit = f'eval_{user_id}_max_edit_p{idx}'

        # create the sbatch file; file name: eval_user_id_pi.sbatch
        res = []
        for is_edit_max in [True, False]:
            sbatch_content = generate_sbatch_content(
                job_name=job_name_max_edit if is_edit_max else job_name_default, 
                email=email, 
                path=path_to_rome,
                start=left, 
                end=right, 
                data_file_path=data_path, 
                run_dir_name=run_dir_name + '_max_edit' if is_edit_max else run_dir_name + "_default_edit",
                is_edit_max=is_edit_max
            )
            res.append(sbatch_content)
        
        # write the sbatch file
        with open(f'{sbatch_tasks_folder}/{job_name_default}.sbatch', 'w') as file:
            file.write(res[1])
        with open(f'{sbatch_tasks_folder}/{job_name_max_edit}.sbatch', 'w') as file:
            file.write(res[0])
        idx += 1

if __name__ == '__main__':
    user_id = int(sys.argv[1])
    create_tasks_for_user(user_id)