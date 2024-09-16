import os, urllib, json, duckdb, logging, shutil, multiprocessing as mp
import urllib.request

def get_data(url, record_count, save_path):
    s3_path = url
    save_name = url.split('/')[-2] + '_' + url.split('/')[-1].split('.')[0]
    if not os.path.exists(f"{save_path}/openalex_raw_data/{save_name}.json"):
        logging.info(f"Downloading data from {url} with {record_count} records...")
        conn = duckdb.connect()
        conn.execute("INSTALL aws; INSTALL httpfs; LOAD aws; LOAD httpfs; LOAD 's3';")
        conn.execute("CREATE SECRET (TYPE S3, PROVIDER CREDENTIAL_CHAIN);")
        conn.execute("SET threads TO 4; SET enable_progress_bar = false;") ## To avoid oom error
        conn.execute("SET temp_directory = '{save_path}/temp';")
        
        # order FROM(identifying the data source) -> WHERE clause (applying filters) ->SELECT clause (choosing which columns to display). 
        # cannot select columns based on the results of a filter
        # because the columns to be returned are determined before the WHERE clause filters are applied.
        query= f"""
                COPY (
                SELECT  id,
                        display_name,
                        publication_year,
                        publication_date,
                        primary_location,
                        open_access,
                        indexed_in,
                        institutions_distinct_count,
                        authorships
                FROM read_json('{s3_path}')
                WHERE   type = 'article' AND
                        is_paratext = false AND
                        is_retracted = false AND
                        publication_year >=2000 AND
                        publication_year <=2023
                ) TO 
                '{save_path}/openalex_raw_data/{save_name}.json' (FORMAT JSON);
                """
        papers = 0
        if record_count > 0:
            try:
                query_result = conn.execute(query).fetchall()
                papers = query_result[0][0]
            except:
                # query returned no result. Might be because filtered out based on filters or some error
                papers = -1
                logging.error(f"Query returned no result for {url}")
        else:
            papers = record_count
        logging.info(f"Downloaded {papers} papers from {url}")
        conn.close()
        return (url, papers)
    else:
        logging.warning(f"Data already exists for {url}. Skipping download...")
        return (url, -1)

def download_openalex(save_path, force_download=False):
    try:
        if os.path.exists(f'{save_path}, openalex_raw_data') and force_download == True:
            logging.warning(f"Data already exists in {save_path}/openalex_raw_data. force_download is set to True. Deleting existing data... ")
            shutil.rmtree(f"{save_path}/openalex_raw_data")
            logging.info(f"Deleted existing data in {save_path}/openalex_raw_data. Downloading fresh data...")
            os.makedirs(f"{save_path}/openalex_raw_data", exist_ok=True)
            urllib.request.urlretrieve("https://openalex.s3.amazonaws.com/data/works/manifest", f"{save_path}/openalex_raw_data/manifest")
            with open(f"{save_path}/openalex_raw_data/manifest", 'r') as file:
                manifest_data = json.load(file)
            url_count_list = [(entry['url'], entry['meta']['record_count'],save_path) for entry in manifest_data['entries']]

            with mp.Pool(processes=16) as pool:
                results = pool.starmap(get_data, url_count_list)
            logging.info("Data download step completed. Creating metadata...")
            urls_parsed_dict = {result[0]: result[1] for result in results}
        elif os.path.exists(f'{save_path}/openalex_raw_data') and force_download == False:
            urls_parsed_dict = {'NA': -1}
            logging.warning(f"Data already exists in {save_path}/openalex_raw_data. force_download is set to False. Skipping download...")
        else:
            logging.info(f"Downloading data from OpenAlex...")
            os.makedirs(f"{save_path}/openalex_raw_data", exist_ok=True)
            urllib.request.urlretrieve("https://openalex.s3.amazonaws.com/data/works/manifest", f"{save_path}/openalex_raw_data/manifest")
            with open(f"{save_path}/openalex_raw_data/manifest", 'r') as file:
                manifest_data = json.load(file)
            url_count_list = [(entry['url'], entry['meta']['record_count'],save_path) for entry in manifest_data['entries']]

            with mp.Pool(processes=16) as pool:
                results = pool.starmap(get_data, url_count_list)
            logging.info("Data download step completed. Creating metadata...")
            urls_parsed_dict = {result[0]: result[1] for result in results}
    except Exception as e:
        urls_parsed_dict = {'NA': -1}
        logging.error(f'Error: {e}')
    return urls_parsed_dict