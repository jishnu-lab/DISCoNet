import os, itertools, subprocess, glob, shutil
from ..utils import _remove_create_folder
import random

def develop_chunks(arguments, max_num_chunks):
    random.shuffle(arguments) # Randomize the order of the arguments for load balancing
    num_cmd_in_chunk = len(arguments)//max_num_chunks + 1
    chunks = [arguments[i:i+num_cmd_in_chunk] for i in range(0, len(arguments), num_cmd_in_chunk)]
    return chunks

def remove_script_files(save_path, job_name):
    script_files = glob.glob(f'{save_path}/slurm_scripts/{job_name}*.sh')
    for script_file in script_files:
        os.remove(script_file)
    return

def step1(arguments, job_name, read_path, save_path, YEARS, shell_script_header_function):
    if len(arguments) >98:
        chunks = develop_chunks(arguments, 98)
    else:
        chunks = [arguments]
    for i, chunk in enumerate(chunks):
        if os.path.exists(f'{save_path}/slurm_scripts/{job_name}_{i}.sh'):
            os.remove(f'{save_path}/slurm_scripts/{job_name}_{i}.sh')
        os.makedirs(f'{save_path}/slurm_outs/{job_name}_{i}', exist_ok=True, mode=0o755)
        with open(f'{save_path}/slurm_scripts/{job_name}_{i}.sh', "w", encoding="utf-8") as f:
            f.write(shell_script_header_function(save_path, job_name = f"{job_name}_{i}" ))
            f.write("\n")
            f.write("commands=(\n")
            for combo in chunk:
                f.write(f"\"python 1_data_downloading.py --parameters \"\\\"\"({YEARS},'{combo}')\"\\\"\" --read_path '{read_path}' --save_path '{save_path}' \" \n ")
            f.write(")\n")
            f.write("\n")
            f.write("eval ${commands[$SLURM_ARRAY_TASK_ID-1]}\n")
            f.write("\ncrc-job-stats\n")
    return

def step2(arguments, job_name, save_path, shell_script_header_function, max_num_chunks=50):
    _remove_create_folder(f'{save_path}/slurm_outs/{job_name}')
    remove_script_files(save_path, job_name)
    chunks = develop_chunks(arguments, max_num_chunks)

    for i, chunk in enumerate(chunks):
        with open(f'{save_path}/slurm_scripts/{job_name}_{i}.sh', "w", encoding="utf-8") as f:
            f.write(shell_script_header_function(save_path, job_name = f"{job_name}" ))
            f.write("\n")
            # f.write("commands=(\n")
            for combo in chunk:
                f.write(f"python 2_embedding_datasets.py --parameters \"{combo}\" --save_path '{save_path}'\n ")
            # f.write(")\n")
            # f.write("\n")
            # f.write("eval ${commands[$SLURM_ARRAY_TASK_ID-1]}\n")
            f.write("\ncrc-job-stats\n")
    return

def step3(arguments, job_name, save_path, shell_script_header_function, max_num_chunks=50):
    _remove_create_folder(f'{save_path}/slurm_outs/{job_name}')
    remove_script_files(save_path, job_name)
    chunks = develop_chunks(arguments, max_num_chunks)
    
    for i, chunk in enumerate(chunks):
        with open(f'{save_path}/slurm_scripts/{job_name}_{i}.sh', "w", encoding="utf-8") as f:
            f.write(shell_script_header_function(save_path, job_name = f"{job_name}" ))
            f.write("\n")
            # f.write("commands=(\n")
            for combo in chunk:
                f.write(f"python 3_preparing_objects_network.py --parameters \"{combo}\" --save_path '{save_path}'\n")
            # f.write(")\n")
            # f.write("\n")
            # f.write("eval ${commands[$SLURM_ARRAY_TASK_ID-1]}\n")
            f.write("\ncrc-job-stats\n")
    return

def step4(arguments, job_name, save_path, shell_script_header_function, max_num_chunks=50):
    _remove_create_folder(f'{save_path}/slurm_outs/{job_name}')
    remove_script_files(save_path, job_name)
    chunks = develop_chunks(arguments, max_num_chunks)

    for i, chunk in enumerate(chunks):
        with open(f'{save_path}/slurm_scripts/{job_name}_{i}.sh', "w", encoding="utf-8") as f:
            f.write(shell_script_header_function(save_path, job_name = f"{job_name}" ))
            f.write("\n")
            # f.write("commands=(\n")
            for combo in chunk:
                if 'zeroshot' in combo:
                    f.write(f"python 4.1_zeroshot.py --parameters \"{combo}\" --save_path '{save_path}'\n")
                elif 'gat' in combo or 'gat_graph_embed' in combo:
                    f.write(f"python 4.2_gat_gat_embed.py --parameters \"{combo}\" --save_path '{save_path}'\n")
            # f.write(")\n")
            # f.write("\n")
            # f.write("eval ${commands[$SLURM_ARRAY_TASK_ID-1]}\n")
            f.write("\ncrc-job-stats\n")
    return

def step5(arguments, job_name, save_path, shell_script_header_function, max_num_chunks=50, **kwargs):
    _remove_create_folder(f'{save_path}/slurm_outs/{job_name}')
    remove_script_files(save_path, job_name)
    chunks = develop_chunks(arguments, max_num_chunks)
    kwargs_str = ' '.join(f"--{key} {value}" for key, value in kwargs.items())  # Convert **kwargs to string
    
    for i, chunk in enumerate(chunks):
        with open(f'{save_path}/slurm_scripts/{job_name}_{i}.sh', "w", encoding="utf-8") as f:
            f.write(shell_script_header_function(save_path, job_name = f"{job_name}" ))
            f.write("\n")
            # f.write("commands=(\n")
            for combo in chunk:
                f.write(f"python 5_plotting.py --parameters \"{combo}\" --save_path '{save_path}'\n")
            # f.write(")\n")
            # f.write("\n")
            # f.write("eval ${commands[$SLURM_ARRAY_TASK_ID-1]}\n")
            f.write("\ncrc-job-stats\n")
        