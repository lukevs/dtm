"""module for tokenizing"""

import stop_words
from spacy.en import English


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
