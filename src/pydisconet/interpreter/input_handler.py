import os, shutil, itertools, logging
from ..utils import _default_inputs, _display_input_help
logger = logging.getLogger(__name__)

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
            logger.info(f"{level} is None")
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
            logger.error(f"Key {key} is not a valid input. Ignoring it and continuing...")
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
        logger.info(f"Consecutive years for GAT is set to {consec_yrs_gat}. This will create all possible gat_X_X+{consec_yrs_gat-1} years")
        year_consec_gat = [f'gat_{year}_{year+consec_yrs_gat-1}' for year in range(start_year, end_year + 2 - consec_yrs_gat)]
        YEARS = YEARS + year_consec_gat

    if (consec_yrs_auth is not None and consec_yrs_auth > 1) :
        logger.info(f"Consecutive years for common authors is set to {consec_yrs_auth}. This will create all possible auth_X_X+{consec_yrs_auth-1} years")
        year_consec_auth = [f'auth_{year}_{year+consec_yrs_auth-1}' for year in range(start_year, end_year + 2 - consec_yrs_auth)]
        YEARS = YEARS + year_consec_auth

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False, mode=0o755)
        
    return process_input, YEARS
