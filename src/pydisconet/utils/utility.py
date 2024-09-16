import gzip, json, pandas as pd, os, itertools, logging, shutil
from io import StringIO
logger = logging.getLogger(__name__)
#####0_processings.py
def _display_input_help():
    print("Expected input parameters and their data types:")
    expected_inputs = {
                        'read_path': ('str', 'Path to the directory where the data is stored'),
                        'save_path': ('str', 'Path to the directory where the data will be saved'),
                        'start_year': ('int', 'Start year to run model from'),
                        'end_year': ('int', 'End year to run model till'),
                        'consec_yrs_gat': ('int', 'Number of consecutive years to consider for GAT'),
                        'consec_yrs_auth': ('int', 'Number of consecutive years to consider for common authors'),
                        'datasets': ('list[str]', "Dataset to run the model on. Choices: ['openalex', 'arxiv']"),
                        'graph_components': ('list[str]', 'Component of the graph to run the model on. Choices: ["full", "lcc"]'),
                        'embedding_modes': ('list[str]', 'Mode of embedding. Choices: ["tfidf", "bert"]'),
                        'models': ('list[str]', 'Model to run the model on. Choices: ["zeroshot", "gat", "gat_graph_embed"]'),
                        'controls': ('list[str]', 'Controls to run the model on. Choices: ["shuffle_y", "shuffle_x", "null"]'),
                        'recreate_folder_level': ('str', "From which level recreate the folders. Choices: ['years', 'datasets', 'graph_components', 'embedding_modes', 'models', 'controls', 'null']")
                        
                    }
    for key, (dtype, description) in expected_inputs.items():
        print(f"- {key} ({dtype}): \n \t{description}")
    return None

#####0_processings.py
def _default_inputs():
    default_inputs = {   
                        'read_path': '/ix/djishnu/Swapnil/coauthorship/Co-Authorship/',
                        'save_path': '/ix/djishnu/Swapnil/coauthorship/Co-Authorship/Results/',
                        'start_year': 2000,
                        'end_year': 2023,
                        'consec_yrs_gat': None,
                        'consec_yrs_auth': None,
                        'datasets': ['openalex','arxiv'],
                        'graph_components': ['full','lcc'],
                        'embedding_modes': ['tfidf','bert'],
                        'models': ['zeroshot','gat','gat_graph_embed'],
                        'controls': ['shuffle_y', 'shuffle_x'],
                        'recreate_folder_level': None,
                    }
    return default_inputs

def _dump_to_json_gz(data, filename):
    if isinstance(data, pd.DataFrame):
        data.to_json(path_or_buf= filename, orient='records') # Convert DataFrame to JSON string
    elif isinstance(data, (list, dict)):
        json_str = json.dumps(data) # Convert list or dict to JSON string
        with gzip.open(filename, 'wt', encoding='utf-8') as file: # Write JSON string to a compressed file
            file.write(json_str)
    else:
        raise ValueError("Data must be a list, dictionary, or pandas DataFrame")
    return None


def _read_from_json_gz(filename):
    # print(filename)
    # with open(filename, 'rb') as f:
    #     magic_number = f.read(2)
    #     print(magic_number) # b'\x1f\x8b'

    with gzip.open(filename, 'rt', encoding='utf-8') as file:
        json_str = file.read()

    # with open(filename, 'rb') as file:
    #     json_str = gzip.GzipFile(fileobj=file)
    #     # json_str = file.read()
    try:
        data = json.loads(json_str) # Try loading as a JSON object (list or dictionary)
    except json.JSONDecodeError:
        json_str_io = StringIO(json_str) # If that fails, try loading as a DataFrame
        data = pd.read_json(json_str_io, orient='records')
    return data

def _remove_create_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=False, mode=0o755)
    return None
    
#####6_user_facing.py
def _check_steps_completion(step_number, path, step_status):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"\tSTEP{step_number}: {path}: Does not exist")
    except FileNotFoundError as e:
        step_status = False
        print(e)
    return step_status

#####6_user_facing.py
def _check_steps(save_path, YEARS, datasets, graph_components=None, embedding_modes=None, models=None, controls=None, step_id=None):
    """
    Checks the completion of steps by verifying the existence of required files.

    Args:
        save_path (str): The base path where files are expected to be found.
        YEARS (list): List of years.
        datasets (list): List of datasets.
        graph_components (list, optional): List of graph components.
        embedding_modes (list, optional): List of embedding modes.
        models (list, optional): List of models.
        controls (list, optional): List of controls.
    """
    if step_id is not None:
        range_list = [step_id]
    else:
        range_list = range(1, 6)
    for i in range_list:
        step_status = True
        if i == 1:
            arguments = itertools.product(YEARS, datasets)
            for arg in arguments:
                path = os.path.join(save_path, arg[0], arg[1], f'{arg[0]}.csv' if 'arxiv' in arg else f'{arg[0]}_journal_filtered.csv')
                step_status = _check_steps_completion(i, path, step_status)
            print(f"STEP{i} CHECK: {'PASSED :-)' if step_status else 'FAILED: Continue only if the files which do not exist should not exist!!!'}")
        
        elif i == 2:
            arguments = itertools.product(YEARS, datasets, graph_components, embedding_modes)
            file_list = ['author_df.json.gz', 'edge_df.json.gz', 'paper_titles.json.gz', 'full_network_properties.csv',
                         'clustering_coefficient.pt', 'degree_centrality.pt', 'degrees.pt',
                         'node_betweenness.pt', 'paper_counts.pt', 'embedded_dataset.pt']
            for arg in arguments:
                for file in file_list:
                    path = os.path.join(save_path, arg[0], arg[1], arg[2], arg[3], file) if file == 'embedded_dataset.pt' \
                            else os.path.join(save_path, arg[0], arg[1], arg[2], file) if file.endswith('.pt') \
                            else os.path.join(save_path, arg[0], arg[1], file)
                    step_status = _check_steps_completion(i, path, step_status)
            print(f"STEP{i} CHECK: {'PASSED :-)' if step_status else 'FAILED: Continue only if the files which do not exist should not exist!!!'}")
        
        elif i == 3:
            arguments = itertools.product(YEARS, datasets, graph_components, embedding_modes, models)
            file_list = ['all_data_object.pt', 'train_data_object.pt', 'val_data_object.pt', 'test_data_object.pt']
            for arg in arguments:
                for file in file_list:
                    path = os.path.join(save_path, arg[0], arg[1], arg[2], arg[3], arg[4], file)
                    path = f"{save_path}/{arg[0]}/{arg[1]}/{arg[2]}/{arg[3]}/{arg[4]}/{file}"
                    step_status = _check_steps_completion(i, path, step_status)
            print(f"STEP{i} CHECK: {'PASSED :-)' if step_status else 'FAILED: Continue only if the files which do not exist should not exist!!!'}")
        
        elif i == 4:
            arguments = itertools.product(YEARS, datasets, graph_components, embedding_modes, models, controls)
            file_list = ['test_df.pkl', 'test_df.pkl', 'all_df.pkl']
            for arg in arguments:
                for idx, file in enumerate(file_list):
                    if idx == 0:
                        path = os.path.join(save_path, arg[0], arg[1], arg[2], arg[3], arg[4], file)
                    elif idx == 1:
                        path = os.path.join(save_path, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], file)
                    else:
                        if 'zeroshot' in arg:
                            path = os.path.join(save_path, arg[0], arg[1], arg[2], arg[3], arg[4], file)
                        else:
                            continue
                    step_status = _check_steps_completion(i, path, step_status)
            print(f"STEP{i} CHECK: {'PASSED :-)' if step_status else 'FAILED: Continue only if the files which do not exist should not exist!!!'}")
        
        elif i == 5:
            arguments = itertools.product(YEARS, datasets)
            file_list = ['topic_model']
            for arg in arguments:
                for file in file_list:
                    path = os.path.join(save_path, 'for_plotting', 'topic_modeling', f"{arg[0]}_{file}")
                    step_status = _check_steps_completion(i, path, step_status)
            print(f"STEP{i} CHECK: {'PASSED :-)' if step_status else 'FAILED: Continue only if the files which do not exist should not exist!!!'}")

        else:
            logger.error(f"STEP{i} check not suported. Exiting...")
    return None

