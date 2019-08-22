############ NOT FINISHED ############

import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats, mark_negation
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('assets/Restaurant_Reviews.tsv',
                      delimiter='\t', quoting=3)
dataset.head()

reviews = list(zip(dataset["Review"], dataset["Liked"]))

train_X, train_Y = zip(*reviews[:800])
test_X, test_Y = zip(*reviews[800:])

analyzer = SentimentAnalyzer()

vocabulary = analyzer.all_words([mark_negation(instance.split())
                                 for instance in train_X[:5000]])
                                     

print("Vocabulary: ", len(vocabulary))

unigram_features = analyzer.unigram_word_feats(vocabulary, min_freq=10)

analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)

print("Unigram Features: ", len(unigram_features))

_train_X = analyzer.apply_features([mark_negation(instance)
                                    for instance in train_X[:5000]])
 
# Build the test set
_test_X = analyzer.apply_features([mark_negation(instance) 
                                   for instance in test_X])
 
trainer = NaiveBayesClassifier.train
classifier = analyzer.train(trainer, list(zip(_train_X, train_Y[:5000])))
 
score = analyzer.evaluate(list(zip(_test_X, test_Y)))
print("Accuracy: ", score['Accuracy'])