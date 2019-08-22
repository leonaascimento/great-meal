import numpy as np
from nltk.corpus import sentiwordnet
from sklearn.naive_bayes import MultinomialNB


class SentiLexiconNB(MultinomialNB):
    def sentiment_ratio(self, X):
        # If we could easyly take the feature names from here we would be able
        # to implement the INB-1 algorithm suggested on the article
        return [1, 1]

    def predict(self, X):
        jll = self.sentiment_ratio(X) * self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
