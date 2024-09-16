import logging, json
from tqdm import tqdm
import dask.bag as db
from datetime import datetime
def flatten(record):
    #Filter year from the latest version of the paper
    datetime_obj = datetime.strptime(record['versions'][-1]['created'], "%a, %d %b %Y %H:%M:%S %Z")
    return {'work_id': record['id'],
            'work_name': record['title'],
            'category':record['categories'],
            'abstract':record['abstract'],
            'year':datetime_obj.year,
            'doi':record["doi"],
            'authors_parsed':record['authors_parsed']}

def process_arxiv(save_path, years_list):
    records=db.read_text(f'{save_path}/arxiv_raw_data/arxiv-metadata-oai-snapshot.json', blocksize='64MB').map(json.loads).map(flatten).to_dataframe()
    individual_years_dict = {int(year): records[records['year'] == int(year)] for year in years_list if '_' not in year}
    papers_parsed_dict = {int(year): None for year in years_list if '_' not in year}
    for year, df in tqdm(individual_years_dict.items(), desc='Saving individual year dataframes for arxiv'):
        computed_df = df.compute()
        computed_df.to_csv(f'{save_path}/{year}/arxiv/{year}.csv', index=False)
        logging.info(f"Data has been written to {save_path}/{year}/arxiv/{year}.csv with paper counts ={len(computed_df)}")
        papers_parsed_dict[year] = len(computed_df)
    return papers_parsed_dict