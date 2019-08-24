import numpy as np
from nltk.corpus import sentiwordnet
from sklearn.naive_bayes import MultinomialNB


class SentiLexiconNB(MultinomialNB):
    def fit(self, X, y, vectorizer, sample_weight=None):
        self.vectorizer = vectorizer
        super().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
