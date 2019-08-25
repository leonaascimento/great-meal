from abc import abstractmethod

import numpy as np
from nltk.corpus import sentiwordnet
from sklearn.naive_bayes import MultinomialNB


class SentimentScore:
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg
        self.delta = pos - neg


class BaseSentiLexiconNB(MultinomialNB):
    @abstractmethod
    def _readjustment_log_likelihood(self, X):
        """Compute the log likelihood readjustment using the senti-lexicon"""

    def fit(self, X, y, feature_names, sample_weight=None):
        self.feature_names = feature_names
        super().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        rll = self._readjustment_log_likelihood(X)
        return self.classes_[np.argmax(rll + jll, axis=1)]


class SentiLexiconNB1(BaseSentiLexiconNB):
    def __init__(self, senti_lexicon, alpha=1.0, fit_prior=True, class_prior=None):
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
        self.senti_lexicon = senti_lexicon

    def _readjustment_log_likelihood(self, X):
        result = []
        for x in X:
            rows, cols = x.nonzero()

            pos = neg = 0
            for col in cols:
                feature = self.feature_names[col]
                score = self.senti_lexicon.get(feature)

                score_delta = score.delta if score is not None else 0

                if score_delta >= 0:
                    pos += 1

                if score_delta <= 0:
                    neg += 1

            rows_size = rows.size
            if not rows_size:
                pos = neg = rows_size = 1

            neg_ratio = np.max([neg/rows_size, 0.0001])
            pos_ratio = np.max([pos/rows_size, 0.0001])

            result.append([neg_ratio, pos_ratio])

        return np.log(result)


class SentiLexiconNB2(BaseSentiLexiconNB):
    def __init__(self, senti_lexicon, alpha=1.0, fit_prior=True, class_prior=None):
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
        self.senti_lexicon = senti_lexicon

    def _readjustment_log_likelihood(self, X):
        result = []
        for x in X:
            _, cols = x.nonzero()

            belonging_cons = np.log(0.9999)
            diverging_cons = np.log(0.0001)

            pos_ratio = neg_ratio = 0
            for col in cols:
                feature = self.feature_names[col]
                score = self.senti_lexicon.get(feature)

                score_delta = score.delta if score is not None else 0

                if score_delta > 0:
                    pos_ratio += belonging_cons
                    neg_ratio += diverging_cons

                if score_delta < 0:
                    pos_ratio += diverging_cons
                    neg_ratio += belonging_cons

            result.append([neg_ratio, pos_ratio])

        return result
