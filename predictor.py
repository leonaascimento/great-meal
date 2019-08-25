import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from naive_bayes import SentiLexiconNB1
from utils import ReviewPreprocessing

dataset = pd.read_csv('assets/Restaurant_Reviews.tsv', delimiter='\t')

preprocessing = ReviewPreprocessing().fit(dataset['Review'])
corpus = preprocessing.corpus
senti_lexicon = preprocessing.senti_lexicon

vectorizing_regex = r"[-_'a-zA-ZÀ-ÖØ-öø-ÿ0-9]+"
vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='word',
                             token_pattern=vectorizing_regex)

classifier = SentiLexiconNB1(senti_lexicon)

X = vectorizer.fit_transform(corpus)
y = dataset.iloc[:, 1].values

classifier.fit(X, y, vectorizer.get_feature_names())

while True:
    review = input('Add your review: ')

    preprocessed_review = preprocessing.transform([review])
    X = vectorizer.transform(preprocessed_review)

    sentiment = 'Positive' if classifier.predict(X)[0] else 'Negative'

    print(sentiment)
