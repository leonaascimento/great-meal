import re

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
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


dataset = pd.read_csv('assets/Restaurant_Reviews.tsv', delimiter='\t')

regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ0-9]+"
lemmatizer = WordNetLemmatizer()

corpus = []
for review in dataset['Review']:
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

pipes = []
pipes.append(('unigram MultinomialNB', make_pipeline(
    CountVectorizer(ngram_range=(1, 1)),
    MultinomialNB())))
pipes.append(('bigram MultinomialNB', make_pipeline(
    CountVectorizer(ngram_range=(2, 2)),
    MultinomialNB())))
pipes.append(('unigram+bigram MultinomialNB', make_pipeline(
    CountVectorizer(ngram_range=(1, 2)),
    MultinomialNB())))

pipes.append(('unigram LinearSVC', make_pipeline(
    CountVectorizer(ngram_range=(1, 1)),
    LinearSVC())))
pipes.append(('bigram LinearSVC', make_pipeline(
    CountVectorizer(ngram_range=(2, 2)),
    LinearSVC())))
pipes.append(('unigram+bigram LinearSVC', make_pipeline(
    CountVectorizer(ngram_range=(1, 2)),
    LinearSVC())))

for title, pipe in pipes:
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall'
    }
    scores = cross_validate(
        pipe, X, y, cv=10, scoring=scoring, return_train_score=False)

    print(f"\nScores for {title}")
    print("  accuracy: %.3f +/- %.3f" %
          (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
    print("  precision: %.3f +/- %.3f" %
          (scores['test_precision'].mean(), scores['test_precision'].std()))
    print("  recall: %.3f +/- %.3f" %
          (scores['test_recall'].mean(), scores['test_recall'].std()))
