import re

from nltk import pos_tag
from nltk.corpus import wordnet, sentiwordnet
from nltk.stem import WordNetLemmatizer


def penn_to_wordnet_tag(penn_tag):
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


class SentimentScore:
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg
        self.delta = pos - neg


class ReviewPreprocessing:
    def __init__(self, lemmatizer=WordNetLemmatizer(), word_pattern=r"[-'a-zA-ZÀ-ÖØ-öø-ÿ0-9]+"):
        self.lemmatizer = lemmatizer
        self.word_pattern = word_pattern
        self.corpus = []
        self.senti_lexicon = {}

    def fit(self, X):
        for x in X:
            words = re.findall(self.word_pattern, x)
            tagged_review = pos_tag(words)

            tokens = []
            for word, penn_tag in tagged_review:
                wn_tag = penn_to_wordnet_tag(penn_tag)
                if wn_tag not in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV):
                    continue

                lemma = self.lemmatizer.lemmatize(word, pos=wn_tag)
                if not lemma:
                    continue

                synsets = wordnet.synsets(lemma, pos=wn_tag)
                if not synsets:
                    continue

                most_common_synset = synsets[0].name()
                senti_synset = sentiwordnet.senti_synset(most_common_synset)

                token = f"{lemma}_{wn_tag[0]}".lower()

                tokens.append(token)
                self.senti_lexicon[token] = SentimentScore(
                    senti_synset.pos_score(),
                    senti_synset.neg_score())

            self.corpus.append(' '.join(tokens))

        return self

    def transform(self, X):
        result = []

        for x in X:
            words = re.findall(self.word_pattern, x)
            tagged_review = pos_tag(words)

            tokens = []
            for word, penn_tag in tagged_review:
                wn_tag = penn_to_wordnet_tag(penn_tag)
                if wn_tag not in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV):
                    continue

                lemma = self.lemmatizer.lemmatize(word, pos=wn_tag)
                if not lemma:
                    continue

                token = f"{lemma}_{wn_tag[0]}".lower()
                tokens.append(token)

            result.append(' '.join(tokens))

        return result
