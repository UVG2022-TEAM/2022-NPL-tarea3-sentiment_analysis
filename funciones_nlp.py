
import pandas as pd
import enum
from typing import Dict, List
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
from spacy.lang.en import English
from gensim.corpora import Dictionary
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class Sentiments(enum.Enum):
    POS = 'POS'
    NEG = 'NEG'


def prepare_data(data_set: pd.DataFrame) -> pd.DataFrame:
    """Transform and cleans the data inside the file path and returns a data frame"""
    data_set['rating'] = data_set['rating'].astype(dtype='int64')
    data_set['sentiment'] = data_set['rating'].apply(lambda x: Sentiments.POS if x >= 40 else Sentiments.NEG)
    return data_set


def predict(review: List[str], model: dict) -> int:       
    positive_log = negative_log = 0
    for word in review[0]:
        if word in model['COND_POS_PROBS']:
            positive_log += model['COND_POS_PROBS'][word]['logprob']
        else:
            positive_log += model['COND_POS_PROBS'][-1]['logprob']
            
        if word in model['COND_NEG_PROBS']:
            negative_log += model['COND_NEG_PROBS'][word]['logprob']
        else:
            negative_log += model['COND_NEG_PROBS'][-1]['logprob']
    if positive_log > negative_log:
        return 1
    return 0  


def sentences_to_words(sentences: List[str]) -> List[List[str]]:
    """Function from list of strings to a list of list of strings"""
    words = []
    for sentence in sentences:
        words.append(simple_preprocess(str(sentence), deacc=True))
    return words


def remove_stopwords(documents: List[List[str]]) -> List[List[str]]:
    """Funtion for removing english stopwords"""
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords.words('english')]
            for doc in documents]


def learn_bigrams(documents: List[List[str]], c=5, t=10) -> List[List[str]]:
    bigram = Phrases(documents, min_count=c, threshold=t)
    bigram_mod = Phraser(bigram)
    return bigram_mod


def create_bigrams(bigram_model, documents: List[List[str]]):
    return [bigram_model[doc] for doc in documents]


def lemmatization(nlp: English, texts: List[List[str]], allowed_postags: List = None) -> List[List[str]]:
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')


def tokenize(documents: List[str], bigram_model) -> List[List[str]]:
    document_words = sentences_to_words(documents)
    document_words = remove_stopwords(document_words)
    document_words = create_bigrams(bigram_model, document_words)
    document_words = lemmatization(nlp, document_words)
    return document_words


def separar_clases(data_set: pd.DataFrame) -> pd.DataFrame:
    data_classes = {sentiment.value: data_set[data_set['sentiment'] == sentiment]['review'].values.tolist() for sentiment in Sentiments}

    positive_reviews = data_classes['POS']
    negative_reviews = data_classes['NEG']

    return positive_reviews, negative_reviews


def create_dictionary(documents: List[List[str]]):
    """Function for creating a dictionary"""
    return Dictionary(documents)


def classified_data(data_set: pd.DataFrame, bigram_model, model: dict) -> pd.DataFrame:
    dfc = data_set.copy()
    result = prepare_data(dfc)
    result['y_pred'] = [0] * len(data_set)
    for i in data_set.index:
        review_words = tokenize([result.review[i]], bigram_model)
        result.at[i, 'y_pred'] = predict(review_words, model)
    result['y_true'] = result['sentiment'].apply(lambda x: 1 if x == Sentiments.POS else 0)
    result.drop('sentiment', inplace=True, axis=1)
    return result


def entrenar_modelos(data_set: pd.DataFrame, bigram_model) -> List[List[str]]:
    positive_reviews, negative_reviews = separar_clases(data_set)

    positive_words = tokenize(positive_reviews, bigram_model)
    negative_words = tokenize(negative_reviews, bigram_model)

    negative_words = [item for sublist in negative_words for item in sublist]
    positive_words = [item for sublist in positive_words for item in sublist]

    dictionary = create_dictionary([negative_words, positive_words])

    negative_bow = dictionary.doc2bow(negative_words)
    positive_bow = dictionary.doc2bow(positive_words)

    total_negative_words = len(negative_words) + len(dictionary)
    total_positive_words = len(positive_words) + len(dictionary)

    negative_word_probs = {}
    for id, count in negative_bow:
        negative_word_probs[dictionary[id]] = {
            'id': id,
            'logprob': np.log((count + 1)/total_negative_words),
        }

    negative_word_probs[-1] = {
        'id': -1,
        'logprob': np.log(1/total_negative_words)
    }

    positive_word_probs = {}
    for id, count in positive_bow:
        positive_word_probs[dictionary[id]] = {
            'id': id,
            'logprob': np.log((count + 1)/total_positive_words),
        }
    positive_word_probs[-1] = {
        'id': -1,
        'logprob': np.log(1/total_positive_words)
    }

    negative_prob = len(negative_words) / (len(negative_words) + len(positive_words))
    positive_prob = len(positive_words) / (len(negative_words) + len(positive_words))

    model = {
        'POS_PROB': np.log(positive_prob),
        'NEG_PROB': np.log(negative_prob),
        'COND_POS_PROBS': positive_word_probs,
        'COND_NEG_PROBS': negative_word_probs
    }

    return model


def calcular_metricas(data_set: pd.DataFrame, ind: int) -> pd.DataFrame:
    prediccion = data_set.y_pred.tolist()
    verdad = data_set.y_true.tolist()
    score_dic = {'bigrama': ind, 
                 'recall': [recall_score(verdad, prediccion, average='binary')],
                 'precision': [precision_score(verdad, prediccion, average='binary')],
                 'f1': [f1_score(verdad, prediccion, average='binary')]}
    return pd.DataFrame.from_dict(score_dic)