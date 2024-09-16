import pandas as pd, pickle, numpy as np, logging, json, gzip
from .plotter_helper import _HexCodeFromFrequencyDict
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from collections import defaultdict
from nltk.corpus import stopwords
from ..analyze import _fit_tf_idf_on_data

# def tokenize_title(paper_titles,tfidf_features):
#     lemma_tokenizer=_LemmaTokenizer()
#     stop_words = set(stopwords.words('english')) 
#     token_stop = lemma_tokenizer(' '.join(stop_words))
#     pipe = Pipeline([('count', CountVectorizer(tokenizer=lemma_tokenizer, stop_words=token_stop, max_features=768)), ('tfid', TfidfTransformer())])
#     pipe.fit(paper_titles)
#     cts = pipe['count'].transform(paper_titles)
#     for key, value in zip(pipe.get_feature_names_out(), np.sum(cts,axis=0).A1 / (1.0*len(paper_titles))):
#         tfidf_features[key]+=(value)
#     return tfidf_features

def tokenize_title(paper_titles,tfidf_features):
    pipe = _fit_tf_idf_on_data(paper_titles)
    cts = pipe['count'].transform(paper_titles)
    for key, value in zip(pipe.get_feature_names_out(), np.sum(cts,axis=0).A1 / (1.0*len(paper_titles))):
        tfidf_features[key]+=(value)
    return tfidf_features

def tokenize_title_per_year(arguments_tuple):
    save_path = arguments_tuple[0]
    year = arguments_tuple[1]
    data= arguments_tuple[2]
    country = arguments_tuple[3]
    save_per_year = arguments_tuple[4]

    print(save_path,year, data, country)
    tfidf_features = defaultdict(float) 
    path = f'{save_path}/{year}/{data}'
    try:
        with gzip.open(f'{path}/author_df.json.gz', 'rt', encoding='utf-8') as f:
            author_df = json.load(f)
        with gzip.open(f'{path}/paper_titles.json.gz', 'rt', encoding='utf-8') as f:
            paper_titles = json.load(f)
        # paper_titles = json.load(open(f'{path}/paper_titles.json.gz', 'r'))
        # author_df = json.load(open(f'{path}/author_df.json.gz', 'r'))
        if country == 'ALL':
            tfidf_features = tokenize_title(paper_titles, tfidf_features)
        else:
            author_df = author_df.explode('work_name')
            author_df = author_df.explode('author_country')
            country_work_dict = author_df.groupby('author_country')['work_name'].apply(set).to_dict()
            paper_titles = list(country_work_dict[country])
            tfidf_features = tokenize_title(paper_titles, tfidf_features)
        if save_per_year:
            pickle.dump(tfidf_features, open(f'{save_path}/combined_year_results/tfidf_features/{country}_{data}_{year}_tfidf_features.pkl', 'wb'))
    
    except Exception as e:
        logging.error(f"Error in reading {year} data for {data} data. Error: {e}")
    return tfidf_features