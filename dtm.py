"""Dynamic Topic Modeling using Non-negative Matrix Factorization"""

import argparse
import logging

import numpy as np
import pandas as pd
import scipy

from models import train_glove, train_tfidf, DTM
from tokens import Tokenizer
from utils import setup_logging, read_data


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

    logger.info('Fitting TF-IDF')
    tokenized_documents = [' '.join(tokens) for tokens in tokenized]
    tfidf = train_tfidf(tokenized_documents)
    vectorized = tfidf.transform(tokenized_documents)

    del tokenized_documents

    logger.info('Training GloVe')
    glove = train_glove(tokenized)

    del tokenized

    logger.info('Separating by windows')
    window_vectors = []
    for window in windows:
        window_slice = np.logical_and(
            documents['timestamp'] >= window['start'],
            documents['timestamp'] < window['end'])

        window_vectors.append(vectorized[np.where(window_slice)])

    logger.info('Fitting topics')
    dtm = DTM(glove, tfidf.get_feature_names())
    dtm.fit(window_vectors)

    logger.info('Done')


if __name__ == '__main__':
    main()
