from .analyze_helpers import _fit_tf_idf_on_data
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from datasets import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
from nltk.corpus import stopwords
from transformers import AutoTokenizer,AutoModelForMaskedLM,AutoModel
import spacy, re, nltk, torch, logging, os, pickle
from tqdm import tqdm
import numpy as np
logger = logging.getLogger(__name__)

# #### TF_IDF embedding code
# class LemmaTokenizer:
#     ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`', '\'', '-']
#     def __init__(self):
#         self.nlp = spacy.load('en_core_web_sm')
#     def __call__(self, doc):
#         doc = re.findall(r"(?u)\b\w\w+\b", doc)
#         doc = re.sub(r"\b\d+\b", " ", " ".join(doc))
        
#         return [token.lemma_ for token in self.nlp(doc) if token.text not in self.ignore_tokens]

#### BERT embedding code
def load_bert_model(save_path):
    cache_dir = f"{save_path}/cache/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    model_name = 'pritamdeka/S-PubMedBert-MS-MARCO'
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name,cache_dir=cache_dir)
    return model, tokenizer

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_batch(texts, model, tokenizer, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    inputs = tokenizer(texts, padding=True, truncation=True,add_special_tokens=True, max_length=512, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**inputs)
    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, inputs['attention_mask']).to(device)
    return sentence_embeddings
    
def stream_data_with_batches(data, model, tokenizer, batch_size=256, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # i=0
    logger.info(f"Running for Batch Size of {batch_size}")
    texts_batch = []
    embedding_final = torch.empty(0, 768).to(device)
    for row in tqdm(data,miniters=int(len(data)*0.1), desc="Data Embedding Progress"):
        # i=i+1
        text_to_embed = str(row['work_name'])
        texts_batch.append(text_to_embed)
    
        if len(texts_batch) == batch_size:
            embeddings_batch = embed_batch(texts_batch, model, tokenizer, device)
            embedding_final = torch.cat((embedding_final, embeddings_batch), dim=0)
            texts_batch = []  # Reset batch for the next iteration
        # if i==100:
        #     print("100")
        #     break
    
    # Process the batch (if it's smaller than the batch size)
    if len(texts_batch) > 0:
        embeddings_batch = embed_batch(texts_batch, model, tokenizer, device)
        embedding_final = torch.cat((embedding_final, embeddings_batch), dim=0)
    logger.info(f"Embedding complete for {len(data)} records")
    return embedding_final

def embedder(save_path, embedding, paper_titles, author_df, batch_size=256, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    if embedding == 'tfidf':
        logger.info(f"Running TF-IDF")
        # nltk.download('stopwords')
        pipe = _fit_tf_idf_on_data(paper_titles)
        # stop_words = set(stopwords.words('english'))
        # lemma_tokenizer=LemmaTokenizer()
        # token_stop = lemma_tokenizer(' '.join(stop_words))
        # pipe = Pipeline([('count', CountVectorizer(tokenizer=lemma_tokenizer, stop_words=token_stop, max_features=768)), ('tfid', TfidfTransformer())])
        # pipe.fit(paper_titles)
        tfidf = pipe.transform(author_df['work_name'])
        # try:
        #     tfidf = pipe.transform(author_df['work_name'])
        # except ValueError:
        #     tfidf = pipe.transform(author_df['work_name'].astype(str))
        embeddings=torch.tensor(tfidf.todense())
    elif embedding == 'bert':
        logger.info(f"Running BERT")
        model, tokenizer = load_bert_model(save_path)
        model = model.to(device)
        embeddings_dataset = Dataset.from_pandas(author_df)
        embeddings=stream_data_with_batches(embeddings_dataset, model, tokenizer, batch_size, device)
    return embeddings
            
def generate_pytorch_data_object(embeddings, edge_df, author_df):
    author_index_dict = dict(zip(author_df.author,range(len(author_df))))
    source_edge_index = list(edge_df.source.map(lambda x: author_index_dict[x]))
    dest_edge_index = list(edge_df.end.map(lambda x: author_index_dict[x]))
    edge_index = torch.tensor([source_edge_index, dest_edge_index], dtype=torch.long)
    embedded_dataset = Data(x=embeddings, edge_index=edge_index,  y=torch.tensor(author_df.loc[:,'work_id'].to_numpy()))
    return author_index_dict, embedded_dataset

def save_pytorch_data_object(path, component, embedded_dataset):
    ud_transform = T.ToUndirected() # Transform to make the edge index undirected
    embedded_dataset = ud_transform(embedded_dataset)
    embedded_dataset.to('cpu')
    if component == 'full':
        torch.save(embedded_dataset,f"{path}/embedded_dataset.pt")
        return embedded_dataset
    else:
        transform = T.LargestConnectedComponents() #Does not reindex. Which is great
        lcc_data = transform(embedded_dataset)
        torch.save(lcc_data,f"{path}/embedded_dataset.pt")
        return lcc_data

def embed_and_save(save_path, component, embedding, paper_titles, edge_df, author_df, batch_size=256, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    path = f"{save_path}/{component}/{embedding}"
    logger.info(f"Embedding for {path}")
    embeddings = embedder(save_path, embedding, paper_titles, author_df, batch_size, device)
    author_index_dict, embedded_dataset = generate_pytorch_data_object(embeddings, edge_df, author_df)
    pickle.dump(author_index_dict, open(f'{save_path}/author_index_dict.pkl', 'wb'))
    data_object = save_pytorch_data_object(path, component, embedded_dataset)
    return data_object
