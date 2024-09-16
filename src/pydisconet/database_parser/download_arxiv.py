import kaggle, logging, os, shutil
def download_arxiv(save_path, force_download=False):
    if os.path.exists(f'{save_path}, arxiv_raw_data') and force_download == True:
        logging.warning(f"Data already exists in {save_path}/arxiv_raw_data. force_download is set to True. Deleting existing data... ")
        shutil.rmtree(f"{save_path}/arxiv_raw_data")
        logging.info(f"Deleted existing data in {save_path}/arxiv_raw_data. Downloading fresh data...")
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('Cornell-University/arxiv', path=f'{save_path}/arxiv_raw_data', unzip=True)
            logging.info('Data downloaded successfully')
        except Exception as e:
            logging.error(f'Error: {e}')
    elif os.path.exists(f'{save_path}/arxiv_raw_data') and force_download == False:
        logging.warning(f"Data already exists in {save_path}/arxiv_raw_data. force_download is set to False. Skipping download...")
    else:
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('Cornell-University/arxiv', path=f'{save_path}/arxiv_raw_data', unzip=True)
            logging.info('Data downloaded successfully')
        except Exception as e:
            logging.error(f'Error: {e}')
    return None