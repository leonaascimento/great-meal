import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv('assets/Restaurant_Reviews.tsv',
                      delimiter='\t', quoting=3)
dataset.head()

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(
        stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).todense()
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

Confusion_Matrix = confusion_matrix(y_test, y_pred)
Accuracy_Score = accuracy_score(y_test, y_pred)

df_cm = pd.DataFrame(Confusion_Matrix, range(2), range(2))
print("Accuracy Score is: ", Accuracy_Score)


def predict(new_review):
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower().split()
    new_review = [ps.stem(word) for word in new_review if word not in set(
        stopwords.words('english'))]
    new_review = ' '.join(new_review)
    new_review = [new_review]
    new_review = cv.transform(new_review).toarray()

    if classifier.predict(new_review)[0] == 1:
        return 'Positive'
    else:
        return 'Negative'


new_review = input('Add a review: ')
while new_review != '':
    feedback = predict(new_review)
    print('This review is: ', feedback)
    new_review = input('Add a review: ')
