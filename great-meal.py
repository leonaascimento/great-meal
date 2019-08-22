import re

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import sentiwordnet, stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB


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
        return wordnet.NOUN


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

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

classifier = MultinomialNB()
scores = cross_val_score(classifier, X, y, cv=10)

print(scores)
print("Accuracy Score is: ", scores.mean())
