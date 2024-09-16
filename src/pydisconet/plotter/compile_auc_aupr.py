import os, logging, torch, pandas as pd, pickle, multiprocessing as mp, itertools, numpy as np, json
from sklearn.metrics import roc_auc_score, average_precision_score
from ..utils import _read_from_json_gz
logging.getLogger(__name__)

def get_auc_aupr(df):
    labels = df['label'].values
    preds = df['pred'].values
    try:
        auc = roc_auc_score(labels, preds)
    except Exception as e:
        logging.error(f"Error in calculating AUC for the data. Error: {e}")
        auc = -1
    try:
        aupr = average_precision_score(labels, preds)
    except Exception as e:
        logging.error(f"Error in calculating AUPR for the data. Error: {e}")
        aupr = -1
    return [auc, aupr]
    # try:
    #     auc = roc_auc_score(df.label, df.pred)
    # except Exception as e:
    #     logging.error(f"Error in calculating AUC for the data. Error: {e}")
    #     auc = -1
    # try:
    #     aupr = average_precision_score(df.label, df.pred)
    # except Exception as e:
    #     logging.error(f"Error in calculating AUPR for the data. Error: {e}")
    #     aupr = -1
    # return [auc, aupr]

def get_quadrants(path, q_logic = 'count'):
    if q_logic == 'count':
        other_metric = torch.load(f"{path}/paper_counts.pt")
    elif q_logic == 'nbw':
        other_metric = torch.load(f"{path}/node_betweenness.pt")
    else:
        raise ValueError('Invalid quadrant logic. It can be either count or nbw.')
    
    other_metric_percentile = np.percentile(other_metric, 95)
    degrees = torch.load(f"{path}/degrees.pt")
    degree_percentile = np.percentile(degrees, 95)
    
    # degrees = torch.load(f"{path}/degrees.pt")
    # degree_percentile = np.percentile(degrees, 95)

    q1_idx = np.where((other_metric <= other_metric_percentile) & (degrees <= degree_percentile))[0]
    q2_idx = np.where((other_metric <= other_metric_percentile) & (degrees > degree_percentile))[0]
    q3_idx = np.where((other_metric > other_metric_percentile) & (degrees <= degree_percentile))[0]
    q4_idx = np.where((other_metric > other_metric_percentile) & (degrees > degree_percentile))[0]
    return q1_idx, q2_idx, q3_idx, q4_idx

def q_auc_aupr(path, df, q_logic = 'count'):
    q1_idx, q2_idx, q3_idx, q4_idx = get_quadrants(path, q_logic)
    q_aucs, q_auprs = [], []


    author0, author1 = df['author0'].values, df['author1'].values
    df['quadrant'] = 'Q'
    
    q1_mask = np.isin(author0, q1_idx) | np.isin(author1, q1_idx)
    q2_mask = np.isin(author0, q2_idx) | np.isin(author1, q2_idx)
    q3_mask = np.isin(author0, q3_idx) | np.isin(author1, q3_idx)
    q4_mask = np.isin(author0, q4_idx) | np.isin(author1, q4_idx)
    
    df.loc[q1_mask, 'quadrant'] = 'Q1'
    df.loc[q2_mask, 'quadrant'] = 'Q2'
    df.loc[q3_mask, 'quadrant'] = 'Q3'
    df.loc[q4_mask, 'quadrant'] = 'Q4'

    for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_df = df[df['quadrant'] == quadrant]
        auc, aupr = get_auc_aupr(q_df)
        q_aucs.append(auc)
        q_auprs.append(aupr)

    # df = df.copy() #to avoid slicing warning in the print. To Much BT
    # df['quadrant'] = 'Q'
    
    # df.loc[df.author0.isin(q1_idx) | df.author1.isin(q1_idx), 'quadrant'] = 'Q1'
    # df.loc[df.author0.isin(q2_idx) | df.author1.isin(q2_idx), 'quadrant'] = 'Q2'
    # df.loc[df.author0.isin(q3_idx) | df.author1.isin(q3_idx), 'quadrant'] = 'Q3'
    # df.loc[df.author0.isin(q4_idx) | df.author1.isin(q4_idx), 'quadrant'] = 'Q4'

    # for i in range(4):
    #     q_df = df[df['quadrant'] == f'Q{i+1}']
    #     auc, aupr = get_auc_aupr(q_df)
    #     q_aucs.append(auc)
    #     q_auprs.append(aupr)
    return [q_logic] + q_aucs + q_auprs

def process_year(arguments_tuple): # (save_path,arguments, country):
    save_path = arguments_tuple[0]
    arguments = arguments_tuple[1]
    country= arguments_tuple[2]
    print(save_path,arguments, country)
    try:
        prediction_df=pd.read_pickle(os.path.join(save_path,*arguments,'test_df.pkl'))
    except Exception as e:
        logging.error(f"Error in reading the test data for {arguments}. Error: {e}")
        return  [arguments] + [country]+ [-1]*2, [arguments] + [country] + [-1]*9, [arguments] + [country] + [-1]*9

    # if country == 'ALL':
    #     logging.info(f"Processing {arguments}")
    #     prediction_df = prediction_df
    if country != 'ALL':
        path = os.path.join(save_path,*arguments[0:2])
        with open(f'{path}/author_index_dict.pkl', 'rb') as f:
            author_index_dict = pickle.load(f)

        # Read the gzip file into a DataFrame
        author_df = pd.read_json(f'{path}/author_df.json.gz')

        # author_df = _read_from_json_gz(f'{path}/author_df.json.gz')
        # with open(f'{path}/author_df.json.gz', 'r') as f:
        #     author_df = json.load(f)
        # author_index_dict = json.load(open(f'{path}/author_index_dict.json.gz', 'r'))
        # author_df = json.load(open(f'{path}/author_df.json.gz', 'r'))
        
        index_author_dict = {v:k for k,v in author_index_dict.items()}                                                            
        # author_df = author_df.explode('work_name')
        # author_df = author_df.explode('author_country')
        author_df = author_df.explode('work_name').explode('author_country')
        country_author_dict = author_df.groupby('author_country')['author'].apply(set).to_dict()

        author1_in_country = prediction_df.author0.astype(int).map(lambda x: index_author_dict[x]).map(lambda x: x in country_author_dict[country])
        author2_in_country = prediction_df.author1.astype(int).map(lambda x: index_author_dict[x]).map(lambda x: x in country_author_dict[country])
        prediction_df = prediction_df[ author1_in_country | author2_in_country]

    if not prediction_df.empty: #if len(prediction_df) > 0:
        path = os.path.join(save_path,*arguments[0:3])
        all_result = [arguments] + [country]+ get_auc_aupr(prediction_df)
        quad_result_nbw = [arguments] + [country] + q_auc_aupr(path, prediction_df, 'nbw')
        quad_result_count = [arguments] + [country] + q_auc_aupr(path, prediction_df, 'count')
    else:
        all_result = [arguments] + [country]+ [-1]*2
        quad_result_nbw = [arguments] + [country] + [-1]*9
        quad_result_count = [arguments] + [country] + [-1]*9
    logging.info(f"Completed {country} {arguments}")
    return all_result, quad_result_nbw, quad_result_count

def calc_auc_aupr_across_years(save_path, data, arguments, country):
    # AUC-AUPR for a year
    all_model_result, all_model_quad_result = [], []
    all_columns_labels = ['arguments','country', 'AUC','AUPR']
    quad_columns_labels = ['arguments', 'country', 'quadrant_logic', 'Q1_AuC', 'Q2_AuC', 'Q3_AuC', 'Q4_AuC', 'Q1_AuPR', 'Q2_AuPR', 'Q3_AuPR', 'Q4_AuPR']
    args_for_map = list(itertools.product([save_path],arguments, [country]))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_year, args_for_map)
    
    for i,result in tqdm(enumerate(results), desc='Joining...'):
        all_model_result.append(result[0])
        all_model_quad_result.append(result[1])
        all_model_quad_result.append(result[2])
    pd.DataFrame(all_model_result, columns=all_columns_labels).to_csv(f'{save_path}/combined_year_results/auc_aupr/{country}_{data}_compiled_results.csv',index=False)
    pd.DataFrame(all_model_quad_result, columns=quad_columns_labels).to_csv(f'{save_path}/combined_year_results/auc_aupr/{country}_{data}_compiled_quad_results.csv',index=False)
    logging.info(f"Completed {country} {data}")
    return None

