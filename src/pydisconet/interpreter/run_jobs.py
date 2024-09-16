import os, itertools, logging, subprocess, glob
from .steps_parser import step1, step2, step3, step4, step5
logger = logging.getLogger(__name__)

def run_job(step, save_path, read_path, shell_script_header_function, max_num_chunks, FINAL_INPUTS, YEARS, datasets, graph_components, embedding_modes, models, controls, **kwargs):
    os.makedirs(f'{save_path}/slurm_scripts/', exist_ok=True, mode=0o755)
    if step == 1:
        job_name = '1_data_downloading'
        arguments = FINAL_INPUTS['datasets']
        step1(arguments, job_name, read_path, save_path, YEARS, shell_script_header_function)
    elif step == 2:
        job_name = '2_embedding_datasets'
        arguments = list(itertools.product(YEARS, datasets, graph_components, embedding_modes,[None], [None]))
        step2(arguments,job_name,save_path,shell_script_header_function,max_num_chunks)
    elif step == 3:
        job_name = '3_preparing_objects_network'
        arguments = list(itertools.product(YEARS, datasets, graph_components, embedding_modes, models, [None]))
        # arguments = arguments + list(itertools.product(YEARS, datasets, graph_components, embedding_modes, models, controls))
        step3(arguments,job_name,save_path,shell_script_header_function, max_num_chunks)
    elif step == 4:
        job_name = '4_running_models'
        arguments = list(itertools.product(YEARS, datasets, graph_components, embedding_modes, models, [None]))
        if controls is not None:
            arguments = arguments + list(itertools.product(YEARS, datasets, graph_components, embedding_modes, models, controls))
        step4(arguments,job_name,save_path,shell_script_header_function, max_num_chunks)
    elif step == 5:
        job_name = '5_compiling_results'
        arguments = list(itertools.product(YEARS, datasets, [None], [None], [None], [None]))
        step5(arguments,job_name,save_path,shell_script_header_function, **kwargs)
    else:
        raise ValueError(f"Step {step} not recognized. Please use 1,2,3,4,5")
    
    script_files = glob.glob(f'{save_path}/slurm_scripts/{job_name}*.sh')
    for script_file in script_files:
        subprocess.run(["sbatch", script_file])
        logger.info(f"Submitted {os.path.basename(script_file)}")