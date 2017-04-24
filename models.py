"""module for models"""

import numpy as np
from glove import Corpus, Glove
from scipy import spatial
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Topic:
    """represents a topic"""

    DEFAULT_TOP_TERMS = 10

    def __init__(self, term_weights, document_weights, vocab):
        """creates a topic

        :param term_weights: term weights for the topic
        :type term_weights: numpy.ndarray

        :param document_weights: document weights for the topic
        :type document_weights: numpy.ndarray

        :param vocab: mapping from index to term
        :type vocab: dict from int to str
        """

        self.term_weights = term_weights
        self.document_weights = document_weights
        self.vocab = vocab

    def top_terms(self, n_top=DEFAULT_TOP_TERMS):
        """gets the top terms of the topic

        :param n_top: number of top terms
        :type n_top: int

        :returns: top term indices
        :rtype: list of int
        """

        top_index = np.argsort(self.term_weights)[::-1][:n_top]
        return [self.vocab[i] for i in top_index]


def print_topics(topics):
    """prints the given topics

    :param topics: topics to print
    :type topics: list of Topic
    """

    for i, t in enumerate(topics):
        print('Topic {}\n'.format(i + 1))
        for word in t.top_words(20):
            print(word)

        print('\n')


class DTM:
    """class implementing Dynamic Topic Models"""

    DEFAULT_COHERENCE_WORDS = 10
    DEFAULT_DYNAMIC_TOP_TERMS = 20

    def __init__(self, glove, vocab):
        """creates a DTM

        :param glove: trained glove model for calculating coherence
        :type glove: glove.Glove

        :param vocab: map from index to word
        :type vocab: dict from int to str
        """

        self.logger = logging.get_logger(__name__)

        self.glove = glove

        self.index_to_word = vocab
        self.word_to_index = {w: i for i, w in enumerate(vocab)}

        self.n_words = len(vocab)

        self.window_topics = []
        self.dynamic_topics = []

    def fit(self, windows):
        """fits the model for the given windows

        :param windows: token matrices for each window
        :type windows: list of np.array
        """

        self.windows = windows
        self._fit_window_topics()
        self._fit_dynamic_topics()

    def get_coherence(self, topic, n_top=DEFAULT_COHERENCE_WORDS):
        """calculates the given topic's coherence

        :param n_top: number of top words to consider in coherence
        :type n_top: int

        :returns: coherence
        :rtype: float
        """

        top_terms = topic.top_terms(n_top)
        word_vecs = list(filter(
            lambda vec: vec is not None,
            (self._get_word_vector(w) for w in top_terms)))

        total_distance = 0
        count = 0

        for i in range(1, len(word_vecs)):
            for j in range(0, i):
                wv_i = word_vecs[i]
                wv_j = word_vecs[j]

                total_distance += 1 - spatial.distance.cosine(wv_i, wv_j)
                count += 1

        return float(total_distance) / (count + 1e-5)

    def _fit_window_topics(self):
        """fits the window topics"""

        self.logger.info('Fitting window topics')
        window_topics = []
        for i, window in enumerate(self.windows):
            self.logger.info('Fitting topic {} of {}'.format(i, len(self.windows))
            window_topics.append(self._choose_topics(window, self.index_to_word))

        self.window_topics = window_topics

    def _fit_dynamic_topics(self, n_top=DEFAULT_DYNAMIC_TOP_TERMS):
        """fits the dynamic topics

        :param n_top: top number of words to consider for dynamic topics
        :type n_top: int
        """

        self.logger.info('Fitting dynamic topics')
        rows = []
        for topics in self.window_topics:
            for topic in topics:
                row = np.zeros((self.n_words,))

                top_word_index = [
                    self.word_to_index[word]
                    for word in topic.top_terms(n_top)
                ]

                row[top_word_index] = topic.term_weights[top_word_index]
                rows.append(row)

        stacked = np.vstack(rows)

        keep_terms = stacked.sum(axis=0) != 0
        keep_term_names = np.array(self.index_to_word)[keep_terms]

        reduced = stacked[:,keep_terms]
        normalized = normalize(reduced, axis=1, norm='l2')

        self.dynamic_topics = self._choose_topics(
            normalized,
            keep_term_names,
            min_n_components=30,
            max_n_components=50)

    def _choose_topics(self, vectors, vocab, min_n_components=10, max_n_components=25):
        """choose the best topics for the given document vectors

        :param vectors: document vectors
        :type vectors: np.ndarray

        :param vocab for the given vectors
        :type vocab: dict from int to str

        :param min_n_components: minimum number of topic components
        :type min_n_components: int

        :param max_n_components: maximum number of topic components
        :type max_n_components: int

        :returns: best topics
        :rtype: list of Topic
        """

        best_coherence = float('-inf')
        best_topics = None

        coherences = []
        for n_components in range(min_n_components, max_n_components + 1):
            w,h = train_nmf(vectors, n_components)
            topics = [
                Topic(term_weights, doc_weights, vocab)
                for term_weights, doc_weights in zip(h, w.T)
            ]

            avg_coherence = (
                sum(self.get_coherence(t) for t in topics) /
                len(topics))

            coherences.append(avg_coherence)

            if avg_coherence > best_coherence:
                best_coherence = avg_coherence
                best_topics = topics

        return best_topics

    def _get_word_vector(self, word):
        """gets the word vector for the given word

        :param word: word to get the vector for
        :type word: str

        :returns: word vector
        :rtype: numpy.ndarray
        """

        return (
            self.glove.word_vectors[self.glove.dictionary[word]]
            if word in self.glove.dictionary
            else None)


def train_nmf(data, n_components):
    """calculates the non-negative matrix factorization (NMF)

    Uses NNDSVD initialization like the paper

    :param data: data to factor
    :type data: numpy.ndarray

    :param n_components: number of factorization components
    :type n_components: int

    :returns: w and h matrix factors
    :rtype: 2-tuple of numpy.ndarray
    """

    model = NMF(n_components=n_components, init='nndsvd')
    w = model.fit_transform(data)
    h = model.components_

    return w,h


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
    glove.fit(corpus.matrix, epochs=20, no_threads=4)
    glove.add_dictionary(corpus.dictionary)

    return glove
