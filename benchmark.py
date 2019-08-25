import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from naive_bayes import BaseSentiLexiconNB, SentiLexiconNB1, SentiLexiconNB2
from utils import ReviewPreprocessing

dataset = pd.read_csv('assets/Restaurant_Reviews.tsv', delimiter='\t')

preprocessing = ReviewPreprocessing().fit(dataset['Review'])
corpus = preprocessing.corpus
senti_lexicon = preprocessing.senti_lexicon

vectorizing_regex = r"[-_'a-zA-ZÀ-ÖØ-öø-ÿ0-9]+"

pipes = []
pipes.append(('SentiLexiconNB1 unigram',
              CountVectorizer(ngram_range=(1, 1), analyzer='word',
                              token_pattern=vectorizing_regex),
              SentiLexiconNB1(senti_lexicon)))
pipes.append(('SentiLexiconNB1 bigram',
              CountVectorizer(ngram_range=(2, 2), analyzer='word',
                              token_pattern=vectorizing_regex),
              SentiLexiconNB1(senti_lexicon)))
pipes.append(('SentiLexiconNB1 unigram+bigram',
              CountVectorizer(ngram_range=(1, 2), analyzer='word',
                              token_pattern=vectorizing_regex),
              SentiLexiconNB1(senti_lexicon)))

pipes.append(('SentiLexiconNB2 unigram',
              CountVectorizer(ngram_range=(1, 1), analyzer='word',
                              token_pattern=vectorizing_regex),
              SentiLexiconNB2(senti_lexicon)))
pipes.append(('SentiLexiconNB2 bigram',
              CountVectorizer(ngram_range=(2, 2), analyzer='word',
                              token_pattern=vectorizing_regex),
              SentiLexiconNB2(senti_lexicon)))
pipes.append(('SentiLexiconNB2 unigram+bigram',
              CountVectorizer(ngram_range=(1, 2), analyzer='word',
                              token_pattern=vectorizing_regex),
              SentiLexiconNB2(senti_lexicon)))

pipes.append(('MultinomialNB unigram',
              CountVectorizer(ngram_range=(1, 1), analyzer='word',
                              token_pattern=vectorizing_regex),
              MultinomialNB()))
pipes.append(('MultinomialNB bigram',
              CountVectorizer(ngram_range=(2, 2), analyzer='word',
                              token_pattern=vectorizing_regex),
              MultinomialNB()))
pipes.append(('MultinomialNB unigram+bigram',
              CountVectorizer(ngram_range=(1, 2), analyzer='word',
                              token_pattern=vectorizing_regex),
              MultinomialNB()))

pipes.append(('LinearSVC unigram',
              CountVectorizer(ngram_range=(1, 1), analyzer='word',
                              token_pattern=vectorizing_regex),
              LinearSVC()))
pipes.append(('LinearSVC bigram',
              CountVectorizer(ngram_range=(2, 2), analyzer='word',
                              token_pattern=vectorizing_regex),
              LinearSVC()))
pipes.append(('LinearSVC unigram+bigram',
              CountVectorizer(ngram_range=(1, 2), analyzer='word',
                              token_pattern=vectorizing_regex),
              LinearSVC()))

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall'
}

y = dataset.iloc[:, 1].values

for title, vectorizer, classifier in pipes:
    start = time.time()
    X = vectorizer.fit_transform(corpus)

    params = None
    if isinstance(classifier, BaseSentiLexiconNB):
        params = {'feature_names': vectorizer.get_feature_names()}

    scores = cross_validate(
        classifier, X, y, cv=10, fit_params=params, scoring=scoring, error_score='raise')

    end = time.time()
    print(f"\nScores for {title}")
    print("  accuracy: %.3f +/- %.3f" %
          (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
    print("  precision: %.3f +/- %.3f" %
          (scores['test_precision'].mean(), scores['test_precision'].std()))
    print("  recall: %.3f +/- %.3f" %
          (scores['test_recall'].mean(), scores['test_recall'].std()))
    print("  execution time: %.6f seg" %
          (end - start))
