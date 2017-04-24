"""module for models"""

from glove import Corpus, Glove
from sklearn.feature_extraction.text import TfidfVectorizer


def train_tfidf(documents):
    """trains tfidf on the tokenized documents

    :param documents: documents
    :type documents: list of str

    :returns: trained model
    :rtype: sklearn.feature_extraction.text.TfidfVectorizer
    """

    tfidf = TfidfVectorizer(norm='l2', max_df=0.2, min_df=5)
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

