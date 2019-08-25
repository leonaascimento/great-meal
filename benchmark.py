import re
import time

import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import sentiwordnet, stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from naive_bayes import BaseSentiLexiconNB, SentiLexiconNB1, SentiLexiconNB2, SentimentScore


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
preprocessing_regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ0-9]+"
lemmatizer = WordNetLemmatizer()

corpus = []
senti_lexicon = {}
for review in dataset['Review']:
    words = re.findall(preprocessing_regex, review)
    tagged_review = pos_tag(words)

    tokens = []
    for word, penn_tag in tagged_review:
        wn_tag = wordnet_tag(penn_tag)
        if wn_tag not in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV):
            continue

        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue

        synsets = wordnet.synsets(lemma, pos=wn_tag)
        if not synsets:
            continue

        most_common_synset = synsets[0].name()
        senti_synset = sentiwordnet.senti_synset(most_common_synset)

        token = f"{lemma}_{wn_tag[0]}".lower()

        tokens.append(token)
        senti_lexicon[token] = SentimentScore(
            senti_synset.pos_score(),
            senti_synset.neg_score())

    corpus.append(' '.join(tokens))


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
