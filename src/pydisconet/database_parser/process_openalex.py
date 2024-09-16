import json, glob, logging, pandas as pd, dask.bag as db, os
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from tqdm import tqdm
from ..utils import _remove_create_folder

def process_data(work):
    try:
        if work['display_name'] is not None:
            # Work Information
            work_id=work['id']
            work_name=work['display_name']
            work_pub_date=work['publication_date']
            work_pub_year=work['publication_year']
            # Source Information
            journal_id=work['primary_location']['source']['id']
            journal_name=work['primary_location']['source']['display_name']
            indexed_in= work['indexed_in']
            is_oa=work['open_access']['is_oa']
            # Author Information
            author_id,author_name,author_country=[],[],[]

            for i in range(len(work['authorships'])):
                if not (len(work['authorships'][i]['countries']) > 0) or not(len(work['authorships'][i]['institutions']) > 0):
                    continue
                else:
                    author_id.append(work['authorships'][i]['author']['id'])
                    author_name.append(work['authorships'][i]['author']['display_name'])
                    if len(work['authorships'][i]['countries']) > 0:
                        author_country.append(work['authorships'][i]['countries'][0])
                    else: ## Using the latest institution as proxy for country
                        author_country=work['authorships'][i]['institutions'][0]['country_code']
                    
            record = {
                    'work_id' : work_id,
                    'work_name' : work_name,
                    'work_pub_date' : work_pub_date,
                    'work_pub_year' : work_pub_year,
                    'journal_id' : journal_id,
                    'journal_name' : journal_name,
                    'author' : author_id,
                    'author_name': author_name,
                    'author_country': author_country,
                    'indexed_in': indexed_in,
                    'is_oa': is_oa
                }
            return record
        else:
            return None
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

def combining_files(input_files, output_file):
    with open(output_file, 'a') as output_handle:
        first_file = True
        for input_file in input_files:
            with open(input_file, 'r') as input_handle:
                if not first_file:
                    next(input_handle)  # Skip the header
                for line in input_handle:
                    output_handle.write(line)
            first_file = False
    return None

def json_loader_filter(record):
    try:
        return json.loads(record)
    except json.decoder.JSONDecodeError:
        return None

def process_openalex(read_path, save_path, years_list):
    logging.info("Starting dask client...")
    _remove_create_folder(f"{save_path}/slurm_outs/1_data_processing_dask")
    cluster = SLURMCluster(
            cores=1,
            memory='10GB',  # Memory per process
            walltime='0-06:00:00',
            account='djishnu',
            job_extra_directives=[  '--job-name=openalex_parsing',
                                    '--cluster=smp',
                                    f'--output={save_path}/slurm_outs/1_data_processing_dask/%A.out',
                                ]
        )
    cluster.adapt(minimum=1, maximum=65)
    client = Client(cluster)

    metadata={
        'work_id': str,
        'work_name': str,
        'work_pub_date': str,
        'work_pub_year': int,
        'journal_id': str,
        'journal_name': str,
        'author': str,
        'author_name': str,
        'author_country': str,
        'indexed_in': str,
        'is_oa': bool
    }

    json_files = glob.glob(f"{save_path}/openalex_raw_data/*.json")
    journals_standardized = pd.read_csv(f"{read_path}/journals_standardized.csv", header=0)
    filter_list = list(journals_standardized['id'])

    processed_records=db.read_text(json_files).map(json_loader_filter).map(process_data)
    filtered_records= processed_records.filter(lambda x: (x is not None) and (x['journal_id'] in filter_list) and (len(x['author'])>0)).persist()  #will trigger computations and keep in DISK memory

    individual_years = [year for year in years_list if '_' not in year]
    papers_parsed_dict = {int(year): -1 for year in years_list if '_' not in year}
    print(individual_years)
    for year in tqdm(individual_years):
        year_data_bag = filtered_records.filter(lambda x: int(x['work_pub_year'])==int(year)).to_dataframe(meta=metadata).to_csv(f"{save_path}/{year}/openalex/{year}_journal_filtered", index=False)
        logging.info(f"Data has been written to {save_path}/{year}/openalex/{year}_journal_filtered")
        input_files = glob.glob(f"{save_path}/{year}/openalex/{year}_journal_filtered/*.part")
        output_file = f'{save_path}/{year}/openalex/{year}_journal_filtered.csv'
        combining_files(input_files, output_file)
        # papers_parsed_dict[int(year)] = len(pd.read_csv(output_file))
        logging.info(f"Data has been written to {save_path}/{year}/openalex/{year}_journal_filtered.csv")

    client.scheduler.shutdown(), client.shutdown(), client.close()
    logging.info("Closing dask client...")
    return papers_parsed_dict