from itertools import combinations
from unidecode import unidecode
from thefuzz import fuzz
import multiprocessing as mp
import pandas as pd
import logging, ast, torch, argparse
import numpy as np
from collections import Counter
logger = logging.getLogger(__name__)

#### Functions related to arxiv data processing
def create_name_work_dict_arxiv(series):
    return dict(zip(series['first_middle'], series['work_id']))

def create_last_name_dict_arxiv(year_filtered_data_raw):
    year_filtered_data = year_filtered_data_raw[['work_id','authors_parsed']].explode('authors_parsed').reset_index(drop=True)
    year_filtered_data['authors_parsed'] = year_filtered_data['authors_parsed'].apply(lambda x: x[:2])
    year_filtered_data[['last','first_middle']] = pd.DataFrame(year_filtered_data['authors_parsed'].to_list(), index=year_filtered_data.index)\
                                                                .map(unidecode)\
                                                                .map(lambda x: x.lower() if isinstance(x, str) else x)\
                                                                
    word_mask = year_filtered_data['last'].str.contains('collaboration|consortium|team', case=False)
    punct_number_mask = year_filtered_data['last'].str.match(r'^[\W\d]$')
    year_filtered_data = year_filtered_data[~word_mask & ~punct_number_mask]
    year_filtered_data[['last','first_middle']] = year_filtered_data[['last','first_middle']].map(lambda x: x.replace('.', '') if isinstance(x, str) else x)
    last_fm_wid = year_filtered_data.groupby(['last','first_middle']).agg({'work_id':list}).reset_index().groupby('last').apply(create_name_work_dict_arxiv).to_dict()
    return  last_fm_wid

def remove_alternatives_arxiv(last, fm_wid):
    if len(fm_wid.keys())>1:
        for first_middle in fm_wid.keys():
            try:
                if len(first_middle) > 1:
                    # print(first_middle)
                    templist = set(fm_wid.keys())-set([first_middle])
                    #### Possible Inital Character alternatives
                    first_middle_split = first_middle.split(' ')
                    initial_characters = [char[0] for char in first_middle_split if len(char) > 0]
                    possible_alternatives = []
                    for k in range(len(initial_characters)):
                        lower_triangle_combination = ""
                        for j in range(k+1):  # +1 to include the diagonal
                            if j == k:
                                lower_triangle_combination += initial_characters[j]
                            else:
                                lower_triangle_combination += initial_characters[j]+" "
                        possible_alternatives.append(lower_triangle_combination)
                    
                    ## Calculating fuzzy word match
                    other_names = templist - set(possible_alternatives)
                    for other_name in other_names:
                            if fuzz.ratio (first_middle, other_name) > 90:
                                possible_alternatives.append(other_name)
                            
                    if len(templist & set(possible_alternatives)) >0:
                        for pa in (templist & set(possible_alternatives)):
                             return (last,first_middle,pa)
            except Exception as e:
                print(e)
                continue
    return None

def create_grouped_df_arxiv(year_filtered_data_raw,last_fm_wid):
    authors = []
    for i in last_fm_wid.keys():
        for j in last_fm_wid[i].keys():
            authors. append((i + " " + j, last_fm_wid[i][j]))

    filtered_data = pd.DataFrame(authors, columns=['author','work_id']).explode('work_id').dropna().reset_index(drop=True)
    filtered_data = filtered_data.merge(year_filtered_data_raw, on='work_id', how='left').drop(columns=['year', 'doi','abstract', 'category', 'authors_parsed'])

    grouped_work = filtered_data.groupby('work_id').agg({
        'work_name': lambda x: list(set(x))[0],
        'author': lambda x: list(combinations(x, 2))}).reset_index()

    grouped_authors = filtered_data.groupby('author').agg({
        'work_name': lambda x: x.to_list(), #','.join(map(str, x)),
        'work_id': lambda x: len(x)}).reset_index()
    return grouped_work, grouped_authors
     
#### Functions related to openalex data processing
def create_grouped_df_openalex(year_filtered_data_raw):
    grouped_work = year_filtered_data_raw[['work_id', 'work_name','author','author_country']]
    grouped_work['author'] = grouped_work['author'].apply(lambda x: list(combinations(x, 2)))
    grouped_work['author_country'] = grouped_work['author_country'].apply(lambda x: list(combinations(x, 2)))
    grouped_authors = year_filtered_data_raw[['work_id', 'work_name','author','author_country']].explode(['author','author_country']).groupby('author').agg({
                                                                                'work_name': lambda x: x.to_list(), #','.join(map(str, x)),
                                                                                'work_id': lambda x: len(x),
                                                                                'author_country': lambda x: list(set(x))}).reset_index()
    return grouped_work, grouped_authors

#### Functions related to both data processing
def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except (SyntaxError, ValueError):
        return []

def generate_grouped_objects(save_path, year, data):
    if data == 'openalex':
        year_filtered_data_raw = pd.read_csv(f'{save_path}/{year}/openalex/{year}_journal_filtered.csv', header=0,converters={'author': safe_literal_eval, 'author_name': safe_literal_eval, 'author_country': safe_literal_eval})
        grouped_work, grouped_author = create_grouped_df_openalex(year_filtered_data_raw)
    elif data == 'arxiv':
        year_filtered_data_raw = pd.read_csv(f'{save_path}/{year}/arxiv/{year}.csv', header=0,converters={'authors_parsed': safe_literal_eval})
        last_fm_wid = create_last_name_dict_arxiv(year_filtered_data_raw)
        with mp.Pool(processes=mp.cpu_count()) as pool:
            index_to_update = pool.starmap(remove_alternatives_arxiv, [(last, fm_wid) for last, fm_wid in last_fm_wid.items()]) 

        for index in index_to_update:
            if index is not None:
                last_fm_wid[index[0]][index[1]] = last_fm_wid[index[0]][index[1]] + last_fm_wid[index[0]][index[2]]
                del last_fm_wid[index[0]][index[2]]
        grouped_work, grouped_author = create_grouped_df_arxiv(year_filtered_data_raw,last_fm_wid)
    
    filtered_grouped_work = (grouped_work[grouped_work['author'].apply(lambda x: len(x) != 0)]).reset_index(drop=True)
    return filtered_grouped_work, grouped_author

def generate_author_edge_df(filtered_work, grouped_author):
    authors_pairs = [item for sublist in filtered_work['author'] for item in sublist]
    authors_count = Counter(authors_pairs)
    graph_df = pd.DataFrame(authors_count.items(), columns=['edge', 'weight'])
    graph_df[['source', 'end']] = pd.DataFrame(graph_df['edge'].to_list(), index=graph_df.index)
    graph_df = graph_df.drop(columns=['edge'])

    logger.debug("Group by sorted 'source' and 'end' treating 'a b' and 'b a' as equivalent")
    graph_df[['source', 'end']] = np.sort(graph_df[['source', 'end']], axis=1)
    edge_df = graph_df.groupby(['source', 'end'])['weight'].sum().reset_index()
    logger.debug("Brief Description of Edges")
    logger.info(f"Initial Unique Edges are {len(edge_df)}")

    logger.debug("Creating Node embeddings")
    author_list=list(set(list(edge_df['source'])+list(edge_df['end'])))    
    author_df=grouped_author[grouped_author['author'].isin(author_list)]
    logger.debug("Brief Description of Nodes")
    logger.info(f"Initial Unique Nodes are {len(author_list)}")
    return edge_df, author_df

def preprocessing_for_embedding(save_path, year, data):
    filtered_grouped_work, grouped_author = generate_grouped_objects(save_path, year, data)
    edge_df, author_df = generate_author_edge_df(filtered_grouped_work, grouped_author)
    return filtered_grouped_work, edge_df, author_df