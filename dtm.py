"""file for running the model"""

import argparse
import json

import pandas as pd
import stop_words
from spacy.en import English


STOP_WORDS = stop_words.get_stop_words('english')


class Tokenizer:
    """class for tokenizing documents"""

    def __init__(self):
        self.nlp = English(tag=True, parse=False, entity=False)

    def tokenize(self, documents, batch_size=1000):
        """tokenize a set of documents

        :param documents: documents to tokenize
        :type documents: list of str

        :param batch_size: batch size for processing documents
        :type batch_size: int

        :returns: tokenized documents (joined by a space)
        :rtype: generator of str
        """

        return (
            ' '.join([token.lemma_ for token in doc if self._include(token)])
            for doc in self.nlp.pipe(
                documents,
                entity=False,
                batch_size=batch_size,
                n_threads=4))

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


def main():
    path = parse_path()
    data = read_data(path)

    documents = pd.DataFrame(data['documents'])
    windows = data['windows']

    tokenizer = Tokenizer()
    tokenized = tokenizer.tokenize(documents['text'])

    print(list(tokenized))


if __name__ == '__main__':
    main()
