import dask, pandas as pd, pickle, os, logging, itertools, argparse,ast
from dask.distributed import Client, as_completed
from dask_jobqueue import SLURMCluster
from tqdm import tqdm
from collections import defaultdict
from plotter import tokenize_title_per_year, process_year

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters', help= 'Parameters', type=str, required=True)
    parser.add_argument('-r', '--cntry_read_path', help= 'Country read path', type=str, required=True)
    parser.add_argument('-spy', '--save_per_year', help= 'Save per year features', type=bool, default=True)
    parser.add_argument('-s', '--save_path', help= 'Save Path', type=str, required=True)
    args = parser.parse_args()
    
    YEARS, data, graph_components, embedding_modes, models, controls = ast.literal_eval(args.parameters)
    cntry_file_path = args.cntry_read_path
    save_per_year = args.save_per_year
    save_path = args.save_path

    # Creating directories for the the analysis
    os.makedirs(f"{save_path}/slurm_outs/5_data_combining_dask", exist_ok=True,mode=0o775)
    if not os.path.exists(f'{save_path}/combined_year_results/tfidf_features'):
        os.makedirs(f'{save_path}/combined_year_results/tfidf_features', exist_ok=False, mode=0o755)

    if not os.path.exists(f'{save_path}/combined_year_results/auc_aupr'):
        os.makedirs(f'{save_path}/combined_year_results/auc_aupr', exist_ok=False, mode=0o755)

    # Setting dask client
    logging.info("Starting dask client...")
    cluster = SLURMCluster(
            cores=1,
            memory='10GB',  # Memory per process
            walltime='0-01:00:00',
            account='djishnu',
            job_extra_directives=[  '--job-name=data_combining',
                                    '--cluster=htc',
                                    f'--output={save_path}/slurm_outs/5_data_combining_dask/%x_%A_%a.out',
                                ]
        )
    cluster.adapt(minimum=1, maximum=64)
    client = Client(cluster)

    # data= datasets[0]
    # country = 'ALL'

    # Reading the countries from the file
    countries=[]
    # cntry_file_path = '/ix/djishnu/Swapnil/coauthorship/Co-Authorship/inputs/countries.txt'
    with open(f'{cntry_file_path}', 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if not line.startswith('#') and len(stripped_line) > 0:
                countries.append(stripped_line)

    # Preparing arguments for AUC_AUPR and TFIDF
    arguments = list(itertools.product(YEARS, [data], graph_components, embedding_modes, models))
    if controls is not None:
        arguments = arguments + list(itertools.product(YEARS, [data], graph_components, embedding_modes, models, controls))

    if (data =='openalex'):
        args_for_map_auc = list(itertools.product([save_path],arguments, countries))
        args_for_map_tfidf = list(itertools.product([save_path], YEARS, [data], countries, [save_per_year]))
    elif (data == 'arxiv'):
        args_for_map_auc = list(itertools.product([save_path],arguments, ['ALL']))
        args_for_map_tfidf = list(itertools.product([save_path], YEARS, [data], ['ALL'], [save_per_year]))
    else:
        logging.error(f"Country {country} not found in the {data}. Skipping...")
        client.close()
        cluster.close()
        exit()
    # Preparing variables for AUC_AUPR
    all_model_result, all_model_quad_result = [], []
    all_columns_labels = ['arguments','country', 'AUC','AUPR']
    quad_columns_labels = ['arguments', 'country', 'quadrant_logic', 'Q1_AuC', 'Q2_AuC', 'Q3_AuC', 'Q4_AuC', 'Q1_AuPR', 'Q2_AuPR', 'Q3_AuPR', 'Q4_AuPR']
    # Preparing variables for TFIDF
    tfidf_features = defaultdict(float)
    # Submit futures with tags and store them in a dictionary
    futures_auc = {client.submit(process_year, *args): 'auc' for args in args_for_map_auc}
    futures_tfidf = {client.submit(tokenize_title_per_year, *args): 'tfidf' for args in args_for_map_tfidf}

    # Combine the futures into a single dictionary
    all_futures = {**futures_auc, **futures_tfidf}

    # Collect results and exceptions with tags
    results_auc, results_tfidf, exceptions = [], [], []

    for future in as_completed(all_futures):
        tag = all_futures[future]
        try:
            result = future.result()
            if tag == 'auc':
                results_auc.append(result)
            else:
                results_tfidf.append(result)
        except Exception as e:
            exceptions.append((future, e))

    #Combining auc_aupr features across years
    for i,result in enumerate(results_auc):
        if (result[0] is not None) and (result[1] is not None) and (result[2] is not None):   
            all_model_result.append(result[0])
            all_model_quad_result.append(result[1])
            all_model_quad_result.append(result[2])
        else:
            continue
    all_model_result = [entry if isinstance(entry, (list, tuple)) else [entry] for entry in all_model_result] #Dealing with NaNs
    all_model_quad_result = [entry if isinstance(entry, (list, tuple)) else [entry] for entry in all_model_quad_result] #Dealing with NaNs
    pd.DataFrame(all_model_result, columns=all_columns_labels).to_csv(f'{save_path}/combined_year_results/auc_aupr/{data}_compiled_results.csv',index=False)
    pd.DataFrame(all_model_quad_result, columns=quad_columns_labels).to_csv(f'{save_path}/combined_year_results/auc_aupr/{data}_compiled_quad_results.csv',index=False)
    logging.info(f"Completed AUC AUPR for {data}")

    #Combining tfidf features across years
    for result in results_tfidf:
        if result is not None:
            for key, value in result.items():
                tfidf_features[key] += value
    pickle.dump(tfidf_features, open(f'{save_path}/combined_year_results/tfidf_features/{data}_tfidf_combined_across_years.pkl', 'wb'))
    logging.info(f"Completed embedding for {data}")
    # Shut down the cluster
    client.close()
    cluster.close()