import string
import nltk as nltk
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
dataset = []
headers = []
with open("news.xml", "r") as file:
    soup = BeautifulSoup(file, "xml")

for news in soup.find_all("news"):
    headers.append(news.find("value", {"name": "head"}).text.strip() + ":")
    tokens = nltk.tokenize.word_tokenize(news.find("value", {"name": "text"}).text.lower())
    lemmatized_words = []
    for word in tokens:
        lemmatized_words.append(lemmatizer.lemmatize(word))
    lemmatized_words = [x for x in lemmatized_words if x not in stopwords.words('english') + list(string.punctuation)]
    nouns_word = [x for x in lemmatized_words if nltk.pos_tag([x])[0][1] == "NN"]
    dataset.append(" ".join(nouns_word))

tfidf_matrix = vectorizer.fit_transform(dataset)
terms = vectorizer.get_feature_names()

for i in range(len(headers)):
    print(headers[i])
    scores = []
    matrix = tfidf_matrix[i].toarray()
    for j in range(len(matrix)):
        for k in range(len(matrix[j])):
            scores.append((matrix[j][k], terms[k]))
    scores = sorted(scores, reverse=True, key=lambda tup: (tup[0], tup[1]))
    print(" ".join([x[1] for x in scores][:5]))
    print()
