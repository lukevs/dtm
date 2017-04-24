"""Dynamic Topic Modeling using Non-negative Matrix Factorization"""

import argparse
import json
import logging.config
import os

import numpy as np
import pandas as pd
import stop_words
from glove import Corpus, Glove
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.en import English


LOGGING_CONFIG_PATH = 'logging.json'
STOP_WORDS = stop_words.get_stop_words('english')


class Tokenizer:
    """class for tokenizing documents"""

    def __init__(self):
        self.nlp = English(tag=True, parse=False, entity=False)

    def tokenize(self, documents, batch_size=1000):
        """tokenize a set of documents

        uses the lemma of each token

        :param documents: documents to tokenize
        :type documents: list of str

        :param batch_size: batch size for processing documents
        :type batch_size: int

        :returns: tokenized documents
        :rtype: list of list of str
        """

        return [
            [token.lemma_ for token in doc if self._include(token)]
            for doc in self.nlp.pipe(
                documents,
                entity=False,
                batch_size=batch_size,
                n_threads=4)
        ]

    @staticmethod
    def _include(token):
        """whether to include a token

        :param token: token to check
        :type token: spacy.tokens.token.Token

        :returns: whether to include
        :rtype: boolean
        """

        return (
            not token.is_punct and
            token.lemma_ not in STOP_WORDS and
            token.lemma_.strip() != '' and
            not token.like_num and
            not token.like_url)


def setup_logging(path=LOGGING_CONFIG_PATH):
    """sets up logging"""

    default_level=logging.INFO
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
            logging.config.dictConfig(config)


def read_data(path):
    """reads data from the path

    :param path: path to data
    :type path: str

    :returns: data
    :rtype: dict
    """

    with open(path, 'r') as f:
        return json.load(f)


def parse_path():
    """parses the path from input arguments

    :returns: path to data file
    :rtype: str
    """

    return parse_args().path


def parse_args():
    """parses dtm args

    :returns: parsed args
    :rtype: Namespace
    """

    parser = argparse.ArgumentParser(description='dynamic topic modeling')
    parser.add_argument(
        'path',
        metavar='PATH',
        type=str,
        help='path to input data file')

    return parser.parse_args()


def train_tfidf(documents):
    """trains tfidf on the tokenized documents

    :param documents: documents
    :type documents: list of str

    :returns: trained model
    :rtype: sklearn.feature_extraction.text.TfidfVectorizer
    """

    tfidf = TfidfVectorizer(
        norm='l2',
        max_df=0.2,
        min_df=5,
        stop_words=STOP_WORDS)

    tfidf.fit(documents)

    return tfidf


def train_glove(tokenized):
    """trains a GloVe model on the tokenized words

    :param tokenized: tokenized documents
    :type tokenized: list of list of str

    :returns: trained model
    :rtype: glove.Glove
    """

    corpus = Corpus()
    corpus.fit(tokenized, window=10)

    glove = Glove(no_components=100, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)

    return glove


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info('Loading data')
    path = parse_path()
    data = read_data(path)
    documents = pd.DataFrame(data['documents'])
    windows = data['windows']

    logger.info('Tokenizing')
    tokenizer = Tokenizer()
    tokenized = tokenizer.tokenize(documents['text'])
    documents['tokenized'] = tokenized

    logger.info('Fitting TF-IDF')
    tokenized_documents = [' '.join(tokens) for tokens in tokenized]
    tfidf = train_tfidf(tokenized_documents)
    documents['vectorized'] = tfidf.transform(tokenized_documents)

    logger.info('Training GloVe')
    glove = train_glove(tokenized)

    logger.info('Separating by windows')
    window_vectors = []
    for window in windows:
        window_slice = np.logical_and(
            documents['timestamp'] >= window['start'],
            documents['timestamp'] < window['end'])

        window_vectors.append(documents[window_slice]['vectorized'])

    logger.info('Done')


if __name__ == '__main__':
    main()
