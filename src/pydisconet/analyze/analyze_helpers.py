import torch, logging, torch.nn.functional as F, spacy, re, nltk
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
logger = logging.getLogger(__name__)

#### Function relating to models
def _scaled_recon_loss(z, pos_edge_index,neg_edge_index):
    EPS = 1e-15 # Epsilon value to prevent log(0)

    pos_loss = -torch.log(torch.clamp((F.cosine_similarity(z[pos_edge_index[0]], z[pos_edge_index[1]], dim=1)*0.5)+0.5,min=0,max=1)+ EPS)
    neg_loss = -torch.log(1-torch.clamp((F.cosine_similarity(z[neg_edge_index[0]], z[neg_edge_index[1]], dim=1)*0.5)+0.5,min=0,max=1)+ EPS) # 1- because of log loss
    batch_loss = pos_loss.mean() + 100*neg_loss.mean()

    return batch_loss

def _sane_loader(data_object, name):
    logger.debug(f"Sanity Check for data loader: {name}")
    assert data_object.edge_label.unique().size(0) == 2 or logger.error("More than 2 labels found") is not None
    assert data_object.edge_label.min() == 0 or logger.error("Negative label not found") is not None
    assert data_object.edge_label.max() == 1 or logger.error("Positive label not found") is not None
    logger.debug(f"Sanity Check for data loader: {name}, Passed!")

    return None

def _sane_reader(train_data, val_data, test_data):
    logger.debug(f"Sanity Check for split_data objects")
    assert len(train_data) != 0 or logger.error("No data found in train data") is not None
    assert len(val_data) != 0 or logger.error("No data found in validation data") is not None
    assert len(test_data) != 0 or logger.error("No data found in test data") is not None
    logger.debug(f"Sanity Check for data objects, Passed!")
    
    return None

def _read_data(path,device):
    all_data = torch.load(f"{path}/all_data_object.pt", map_location=device)
    train_data = torch.load(f"{path}/train_data_object.pt", map_location=device)
    val_data = torch.load(f"{path}/val_data_object.pt", map_location=device)
    test_data = torch.load(f"{path}/test_data_object.pt", map_location=device)
    _sane_reader(train_data, val_data, test_data)

    return all_data, train_data, val_data, test_data

class _LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`', '\'', '-']
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm') ### Already does lemmatization
    def __call__(self, doc):
        doc = re.findall(r"(?u)\b\w\w+\b", doc)
        doc = re.sub(r"\b\d+\b", " ", " ".join(doc))
        return [token.lemma_ for token in self.nlp(doc) if not token.is_stop and not token.is_punct and not re.match(r'^\s*$', token.text) \
                and token.text not in self.ignore_tokens and len(token.text) > 2]

def _fit_tf_idf_on_data(data):
    nltk.download('stopwords')
    lemma_tokenizer = _LemmaTokenizer()
    stop_words = set(stopwords.words('english'))
    token_stop = lemma_tokenizer(' '.join(stop_words))
    pipe = Pipeline([('count', CountVectorizer(tokenizer=lemma_tokenizer, stop_words=token_stop, max_features=768, max_df = 0.995, min_df = 10)), ('tfidf', TfidfTransformer())])
    pipe.fit(data)

    return pipe