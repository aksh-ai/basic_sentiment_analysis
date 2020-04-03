from __future__ import print_function, division
from future.utils import iteritems
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

def tokenizer(sentence):
    sentence = sentence.lower()
    tokens = nltk.tokenize.word_tokenize(sentence)
    tokens = [token for token in tokens if len(token)>2]
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

def vectorizer(tokens, labels):
    x = np.zeros(len(index_map)+1)
    for token in tokens:
        index = index_map[token]
        x[index] += 1
    x = x/x.sum()
    x[-1] = labels
    return x    

wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews = BeautifulSoup(open('dataset/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('dataset/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(positive_reviews)]

index_map = {}
current_index = 0
positive_tokens = []
negative_tokens = []

for review in positive_reviews:
    tokens = tokenizer(review.text)
    positive_tokens.append(tokens)
    for token in tokens:
        if token not in index_map:
            index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = tokenizer(review.text)
    negative_tokens.append(tokens)
    for token in tokens:
        if token not in index_map:
            index_map[token] = current_index
            current_index += 1

N = len(positive_tokens) + len(negative_tokens)

data = np.zeros((N, len(index_map)+1))

i = 0

for token in positive_tokens:
    d = vectorizer(token, 1)
    data[i, :] = d
    i += 1 

for token in negative_tokens:
    d = vectorizer(token, 0)
    data[i, :] = d
    i += 1 

np.random.shuffle(data)

X = data[:, :-1]
y = data[:, -1]

X_train = X[:-100, ]
y_train = y[:-100, ]
X_test = X[-100:,]
y_test = y[-100:,]

model = LogisticRegression()

model.fit(X_train, y_train)

print("Accuracy Score {:.2f} %\n".format(model.score(X_test, y_test)*100))

print("Words and their weights: \n")

treshold = 0.8

for word, index in iteritems(index_map):
    weight = model.coef_[0][index]
    if weight > treshold or weight < -treshold:
        t = "negative" if weight<0 else "positive"
        print(f"{word}: {weight:.4f} , {t}")

print()