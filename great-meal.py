import re

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from naive_bayes import SentiLexiconNB


def wordnet_tag(penn_tag):
    if penn_tag.startswith('J'):
        return wordnet.ADJ
    elif penn_tag.startswith('V'):
        return wordnet.VERB
    elif penn_tag.startswith('N'):
        return wordnet.NOUN
    elif penn_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def default_treatment(reviews):
    regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ0-9]+"
    stemmer = PorterStemmer()

    corpus = []
    for review in reviews:
        words = re.findall(regex, review)
        stems = [stemmer.stem(word) for word in words if not word in set(
            stopwords.words('english'))]

        corpus.append(' '.join(stems))

    X = corpus
    y = dataset.iloc[:, 1].values

    return (X, y)


def improved_treatment(reviews):
    regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ0-9]+"
    lemmatizer = WordNetLemmatizer()

    corpus = []
    for review in reviews:
        words = re.findall(regex, review)
        tagged_review = pos_tag(words)

        lemmas = []
        for word, penn_tag in tagged_review:
            tag = wordnet_tag(penn_tag)

            if tag in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV):
                lemma = lemmatizer.lemmatize(word, pos=tag)
                lemmas.append(f'{lemma}_{penn_tag[0]}')

        corpus.append(' '.join(lemmas))

    X = corpus
    y = dataset.iloc[:, 1].values

    return (X, y)


pipes = []
pipes.append(('SentiLexiconNB unigram',
              CountVectorizer(ngram_range=(1, 1)), SentiLexiconNB()))
pipes.append(('SentiLexiconNB bigram',
              CountVectorizer(ngram_range=(2, 2)), SentiLexiconNB()))
pipes.append(('SentiLexiconNB unigram+bigram',
              CountVectorizer(ngram_range=(1, 2)), SentiLexiconNB()))

pipes.append(('MultinomialNB unigram',
              CountVectorizer(ngram_range=(1, 1)), MultinomialNB()))
pipes.append(('MultinomialNB bigram',
              CountVectorizer(ngram_range=(2, 2)), MultinomialNB()))
pipes.append(('MultinomialNB unigram+bigram',
              CountVectorizer(ngram_range=(1, 2)), MultinomialNB()))

pipes.append(('MultinomialNB unigram',
              CountVectorizer(ngram_range=(1, 1)), LinearSVC()))
pipes.append(('MultinomialNB bigram',
              CountVectorizer(ngram_range=(2, 2)), LinearSVC()))
pipes.append(('MultinomialNB unigram+bigram',
              CountVectorizer(ngram_range=(1, 2)), LinearSVC()))

dataset = pd.read_csv('assets/Restaurant_Reviews.tsv', delimiter='\t')

X1, y1 = default_treatment(dataset['Review'])
X2, y2 = improved_treatment(dataset['Review'])

corpora = []
corpora.append(('default treatment', X1, y1))
corpora.append(('improved treatment', X2, y2))

for title, vectorizer, classifier in pipes:
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall'
    }

    for treatment, corpus, labels in corpora:
        X = vectorizer.fit_transform(corpus)
        y = labels

        params = None

        if isinstance(classifier, SentiLexiconNB):
            params = {'vectorizer': vectorizer}

        scores = cross_validate(
            classifier, X, y, cv=10, fit_params=params, scoring=scoring)

        print(f"\nScores for {title} with {treatment}")
        print("  accuracy: %.3f +/- %.3f" %
              (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
        print("  precision: %.3f +/- %.3f" %
              (scores['test_precision'].mean(), scores['test_precision'].std()))
        print("  recall: %.3f +/- %.3f" %
              (scores['test_recall'].mean(), scores['test_recall'].std()))
