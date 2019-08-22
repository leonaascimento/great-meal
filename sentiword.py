import re
import nltk
import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats, mark_negation
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import accuracy_score

lemmatizer = WordNetLemmatizer()

def tag_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def predict(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = pos_tag(word_tokenize(review))

    tokens_count = 0
    sentiment = 0.0

    for word, tag in review:
        wn_tag = tag_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue

        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue

        synsets = wn.synsets(lemma, pos=wn_tag)

        if not synsets:
            continue
        
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        tokens_count += 1

    if not tokens_count:
        return 0

    if sentiment >= 0:
        return 1
 
    return 0

dataset = pd.read_csv('assets/Restaurant_Reviews.tsv',
                      delimiter='\t', quoting=3)
dataset.head()

corpus = dict([])
corpus['Review'] = []
corpus['Liked'] = []

for index, row in dataset.iterrows():
    rev = re.sub('[^a-zA-Z]', ' ', row['Review'])
    rev = rev.lower()
    rev = rev.split()

    rev = [lemmatizer.lemmatize(word) for word in rev if not word in set(
        stopwords.words('english'))]

    rev = ' '.join(rev)
    corpus['Review'].append(rev)
    corpus['Liked'].append(row['Liked'])

reviews = list(zip(corpus['Review'], corpus['Liked']))

test_X, test_Y = zip(*reviews[:1000])

pred_y = [predict(text) for text in test_X]
print(accuracy_score(test_Y, pred_y))

new_review = input('Add a review: ')
while new_review != '':
    feedback = predict(new_review)
    print('This review is: ', feedback)
    new_review = input('Add a review: ')
