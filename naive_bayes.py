import numpy as np
from nltk.corpus import sentiwordnet
from sklearn.naive_bayes import MultinomialNB


class SentimentScore:
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg
        self.delta = pos - neg


class SentiLexiconNB(MultinomialNB):
    def __init__(self, senti_lexicon, alpha=1.0, fit_prior=True, class_prior=None):
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
        self.senti_lexicon = senti_lexicon

    def _log_ratio_of_pattern(self, X):
        rows, cols = X.nonzero()

        pos = neg = 0
        for col in cols:
            feature = self.feature_names[col]
            score = self.senti_lexicon.get(feature)

            score_delta = score.delta if score is not None else 0

            if score_delta >= 0:
                pos += 1

            if score_delta <= 0:
                neg += 1

        neg_ratio = np.max([neg/rows.size, 0.0001])
        pos_ratio = np.max([pos/rows.size, 0.0001])

        return np.log([neg_ratio, pos_ratio])

    def fit(self, X, y, feature_names, sample_weight=None):
        self.feature_names = feature_names
        super().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        jll = self._log_ratio_of_pattern(X) + self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
