from utils import _default_inputs, _display_input_help
import os, itertools, shutil, subprocess, ast, yaml, logging, glob, torch, argparse

def process_restart(recreate_folder_level=None, save_path=None, YEARS=None, datasets=None, graph_components=None, embedding_modes=None, models=None, controls=None):
    if recreate_folder_level is not None:
        recreate_folder_level = recreate_folder_level[0]
        print(f"Restart values found as: {recreate_folder_level}. Recreating the paths for the restart values")
        if recreate_folder_level == 'years':
            paths = itertools.product(YEARS)
        elif recreate_folder_level == 'datasets':
            paths = itertools.product(YEARS, datasets)
        elif recreate_folder_level == 'graph_components':
            paths = itertools.product(YEARS, datasets, graph_components)
        elif recreate_folder_level == 'embedding_modes':
            paths = itertools.product(YEARS, datasets, graph_components, embedding_modes)
        elif recreate_folder_level == 'models':
            paths = itertools.product(YEARS, datasets, graph_components, embedding_modes, models)
        elif recreate_folder_level == 'controls':
            paths = itertools.product(YEARS, datasets, graph_components, embedding_modes, models, controls)
        else:
            raise ValueError(f"Restart level: {recreate_folder_level} not recognized. Please use 'years', 'datasets', 'graph_components', 'embedding_modes', 'models', 'controls'")
        
        for comb in paths:
            path = os.path.join(save_path, *comb)
            if os.path.exists(path):
                shutil.rmtree(path)
            print(path)
           
def check_list_inputs(level, actual_input, allowed_input):
    if actual_input is None:
        if level not in ['controls', 'restart_level']:
            raise ValueError(f"Input is None. Please use ATLEAST ONE of the following: {allowed_input}")
        else :
            logging.info(f"{level} is None")
    else: 
        for item in actual_input:
            if item not in allowed_input:
                raise ValueError(f"Input {item} not recognized. Please use ATLEAST ONE of the following: {allowed_input}") 
    return None

def process_inputs(user_inputs):
    process_input = _default_inputs()
    for key, value in user_inputs.items():
        try:
            process_input[key] = value
        except KeyError:
            logging.error(f"Key {key} is not a valid input. Ignoring it and continuing...")
            _display_input_help()
        
    consec_yrs_gat = process_input['consec_yrs_gat']
    consec_yrs_auth = process_input['consec_yrs_auth']
    start_year = process_input['start_year']
    end_year = process_input['end_year']
    save_path = process_input['save_path']


    check_list_inputs('datasets',process_input['datasets'], ['openalex','arxiv'])
    check_list_inputs('graph_components',process_input['graph_components'], ['full','lcc'])
    check_list_inputs('embedding_modes',process_input['embedding_modes'], ['tfidf','bert'])
    check_list_inputs('models',process_input['models'], ['zeroshot','gat','gat_graph_embed'])
    check_list_inputs('controls',process_input['controls'], ['shuffle_y','shuffle_x', None])
    check_list_inputs('recreate_folder_level',process_input['recreate_folder_level'], ['years', 'datasets', 'graph_components', 'embedding_modes', 'models', 'controls', None])
    
    
    if (consec_yrs_gat is not None and (consec_yrs_gat > end_year - start_year + 1)) or (consec_yrs_auth is not None and (consec_yrs_auth > end_year - start_year + 1)):
        raise ValueError("Number of consecutive years for GAT/ common authors cannot be more than the total number of years.")

    YEARS = [str(year) for year in range(start_year, end_year+1)]
    if (consec_yrs_gat is not None and consec_yrs_gat > 1) :
        logging.info(f"Consecutive years for GAT is set to {consec_yrs_gat}. This will create all possible gat_X_X+{consec_yrs_gat-1} years")
        year_consec_gat = [f'gat_{year}_{year+consec_yrs_gat-1}' for year in range(start_year, end_year + 2 - consec_yrs_gat)]
        YEARS = YEARS + year_consec_gat

    if (consec_yrs_auth is not None and consec_yrs_auth > 1) :
        logging.info(f"Consecutive years for common authors is set to {consec_yrs_auth}. This will create all possible auth_X_X+{consec_yrs_auth-1} years")
        year_consec_auth = [f'auth_{year}_{year+consec_yrs_auth-1}' for year in range(start_year, end_year + 2 - consec_yrs_auth)]
        YEARS = YEARS + year_consec_auth

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False, mode=0o755)
        
    return process_input, YEARS


def shell_script_cpu(commands, save_path, job_name):
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --cluster=htc
#SBATCH --array=1-{str(len(commands))}
#SBATCH -t 2-00:00:00
#SBATCH --output={save_path}/slurm_outs/{job_name}/%x_%A_%a.out
#SBATCH --ntasks={2 if '5_' in job_name else 16}
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
##SBATCH --mail-user=swk25@pitt.edu
##SBATCH --mail-type=END,FAIL

# module purge
# module load python/ondemand-jupyter-python3.10
# source activate /ix/djishnu/Swapnil/.conda/envs/coauth_env/

"""
    return sbatch_script

def shell_script_gpu(commands, save_path, job_name):
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={1 if '3_' in job_name else 4}
#SBATCH --mem-per-cpu=40G ### Will allocate 40*4 to each gpu ## To preven cuda oom
#SBATCH --partition=l40s,a100,gtx1080,a100_nvlink
#SBATCH --array=1-{str(len(commands))}
#SBATCH -t 2-00:00:00
#SBATCH --output={save_path}/slurm_outs/{job_name}/%x_%A_%a.out
##SBATCH --mail-user=swk25@pitt.edu
##SBATCH --mail-type=END,FAIL

# module purge
# module load python/ondemand-jupyter-python3.10
# source activate /ix/djishnu/Swapnil/.conda/envs/coauth_env/

"""
    return sbatch_script

def develop_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    return lst

def step1(arguments, job_name):
    chunks = develop_chunks(arguments, 500) if len(arguments) > 500 else [arguments]
    list(map(os.remove, glob.glob(f'{save_path}/slurm_scripts/{job_name}*.sh')))

    for i, chunk in enumerate(chunks):
        os.makedirs(f'{save_path}/slurm_outs/{job_name}_{i}', exist_ok=True, mode=0o755)
        with open(f'{save_path}/slurm_scripts/{job_name}_{i}.sh', "w") as f:
            f.write(shell_script_cpu(chunk, f"{save_path}", job_name = f"{job_name}_{i}" ))
            f.write("\n")
            f.write("commands=(\n")
            for combo in chunk:
                f.write(f"\t \"python 1_data_downloading.py --parameters \"\\\"\"({YEARS},'{combo}')\"\\\"\" --read_path '{read_path}' --save_path '{save_path}' \" \n ")
            f.write(")\n")
            f.write("\n")
            f.write("eval ${commands[$SLURM_ARRAY_TASK_ID-1]}\n")
            f.write("\ncrc-job-stats\n")
    return

def step2(arguments, job_name):
    chunks = develop_chunks(arguments, 500) if len(arguments) > 500 else [arguments]
    list(map(os.remove, glob.glob(f'{save_path}/slurm_scripts/{job_name}*.sh')))

    for i, chunk in enumerate(chunks):
        os.makedirs(f'{save_path}/slurm_outs/{job_name}_{i}', exist_ok=True, mode=0o755)
        with open(f'{save_path}/slurm_scripts/{job_name}_{i}.sh', "w") as f:
            f.write(shell_script_gpu(chunk, f"{save_path}", job_name = f"{job_name}_{i}" ))
            f.write("\n")
            f.write("commands=(\n")
            for combo in chunk:
                f.write(f"\t \"python 2_embedding_datasets.py --parameters \"\\\"\"{combo}\"\\\"\" --save_path '{save_path}' \" \n ")
            f.write(")\n")
            f.write("\n")
            f.write("eval ${commands[$SLURM_ARRAY_TASK_ID-1]}\n")
            f.write("\ncrc-job-stats\n")
    return

def step3(arguments, job_name):
    chunks = develop_chunks(arguments, 500) if len(arguments) > 500 else [arguments]
    list(map(os.remove, glob.glob(f'{save_path}/slurm_scripts/{job_name}*.sh')))

    for i, chunk in enumerate(chunks):
        os.makedirs(f'{save_path}/slurm_outs/{job_name}_{i}', exist_ok=True, mode=0o755)
        with open(f'{save_path}/slurm_scripts/{job_name}_{i}.sh', "w") as f:
            f.write(shell_script_gpu(chunk, f"{save_path}", job_name = f"{job_name}_{i}" ))
            f.write("\n")
            f.write("commands=(\n")
            for combo in chunk:
                f.write(f"\t \"python 3_preparing_objects_network.py --parameters \"\\\"\"{combo}\"\\\"\" --save_path '{save_path}' \" \n ")
            f.write(")\n")
            f.write("\n")
            f.write("eval ${commands[$SLURM_ARRAY_TASK_ID-1]}\n")
            f.write("\ncrc-job-stats\n")
    return

def step4(arguments, job_name):
    chunks = develop_chunks(arguments, 500) if len(arguments) > 500 else [arguments]
    list(map(os.remove, glob.glob(f'{save_path}/slurm_scripts/{job_name}*.sh')))

    for i, chunk in enumerate(chunks):
        os.makedirs(f'{save_path}/slurm_outs/{job_name}_{i}', exist_ok=True, mode=0o755)
        with open(f'{save_path}/slurm_scripts/{job_name}_{i}.sh', "w") as f:
            f.write(shell_script_gpu(chunk, f"{save_path}", job_name = f"{job_name}_{i}" ))
            f.write("\n")
            f.write("commands=(\n")
            for combo in chunk:
                if 'zeroshot' in combo:
                    f.write(f"\t \"python 4.1_zeroshot.py --parameters \"\\\"\"{combo}\"\\\"\" --save_path '{save_path}' \" \n ")
                elif 'gat' in combo or 'gat_graph_embed' in combo:
                    f.write(f"\t \"python 4.2_gat_gat_embed.py --parameters \"\\\"\"{combo}\"\\\"\" --save_path '{save_path}' \" \n ")
            f.write(")\n")
            f.write("\n")
            f.write("eval ${commands[$SLURM_ARRAY_TASK_ID-1]}\n")
            f.write("\ncrc-job-stats\n")
    return

def step5(arguments, job_name):
    chunks = develop_chunks(arguments, 500) if len(arguments) > 500 else [arguments]
    list(map(os.remove, glob.glob(f'{save_path}/slurm_scripts/{job_name}*.sh')))

    for i, chunk in enumerate(chunks):
        os.makedirs(f'{save_path}/slurm_outs/{job_name}_{i}', exist_ok=True, mode=0o755)
        with open(f'{save_path}/slurm_scripts/{job_name}_{i}.sh', "w", encoding="utf-8") as f:
            f.write(shell_script_cpu(chunk, f"{save_path}", job_name = f"{job_name}_{i}" ))
            f.write("\n")
            f.write("commands=(\n")
            for combo in chunk:
                f.write(f"\t \"python 5_plotting.py --parameters \"\\\"\"{combo}\"\\\"\" --cntry_read_path '/ix/djishnu/Swapnil/coauthorship/Co-Authorship/inputs/countries.txt' --save_path '{save_path}' \" \n ")
            f.write(")\n")
            f.write("\n")
            f.write("eval ${commands[$SLURM_ARRAY_TASK_ID-1]}\n")
            f.write("\ncrc-job-stats\n")
        

def run_job(step):
    os.makedirs(f'{save_path}/slurm_scripts/', exist_ok=True, mode=0o755)
    if step == 1:
        job_name = '1_data_downloading'
        paths = itertools.product(YEARS, datasets, graph_components, embedding_modes, models, controls)
        for path in paths:
            os.makedirs(os.path.join(save_path, *path), exist_ok=True, mode=0o755)
        arguments = FINAL_INPUTS['datasets']
        step1(arguments,job_name)
    elif step == 2:
        job_name = '2_embedding_datasets'
        arguments = list(itertools.product(YEARS, datasets, graph_components, embedding_modes,[None], [None]))
        step2(arguments,job_name)
    elif step == 3:
        job_name = '3_preparing_objects_network'
        # arguments = list(itertools.product(YEARS, datasets, graph_components, [embedding_modes[0]], [None], [None]))
        # arguments = arguments + list(itertools.product(YEARS, datasets, graph_components, embedding_modes, models, controls))
        arguments = list(itertools.product(YEARS, datasets, graph_components, embedding_modes, models, [None]))
        step3(arguments,job_name)
    elif step == 4:
        job_name = '4_running_models'
        arguments = list(itertools.product(YEARS, datasets, graph_components, embedding_modes, models, [None]))
        arguments = arguments + list(itertools.product(YEARS, datasets, graph_components, embedding_modes, models, controls))
        step4(arguments,job_name)
    elif step == 5:
        job_name = '5_compiling_results'
        arguments = [(YEARS, data, graph_components, embedding_modes, models, controls) for data in FINAL_INPUTS['datasets']]
        step5(arguments,job_name)
    else:
        raise ValueError(f"Step {step} not recognized. Please use 1,2,3,4")
    
    script_files = glob.glob(f'{save_path}/slurm_scripts/{job_name}*.sh')
    for script_file in script_files:
        subprocess.run(["sbatch", script_file])
        logging.info(f"Submitted {script_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_yaml_path', help= 'Input Yaml Path', type=str, required=True)
    parser.add_argument('-st', '--step', help= 'run_step_number', type=str, required=True)
    _display_input_help()

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_path = args.input_yaml_path
    step = args.step

    with open(input_path, 'r') as file:
        user_inputs = yaml.safe_load(file)

    FINAL_INPUTS, YEARS = process_inputs(user_inputs)
    read_path = FINAL_INPUTS['read_path']
    save_path =FINAL_INPUTS['save_path']
    years = YEARS
    datasets = FINAL_INPUTS['datasets']
    graph_components = FINAL_INPUTS['graph_components']
    embedding_modes = FINAL_INPUTS['embedding_modes']
    models = FINAL_INPUTS['models']
    controls = FINAL_INPUTS['controls']
    if FINAL_INPUTS['recreate_folder_level'][0] is not None:
        process_restart(FINAL_INPUTS['recreate_folder_level'], save_path,years, datasets, graph_components, embedding_modes, models, controls)
        for comb in all_combinations:
            path = os.path.join(save_path, *comb)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=False, mode=0o755)

    logging.info(f"Running step {step} for the following inputs: {FINAL_INPUTS}")
    run_job(int(step))