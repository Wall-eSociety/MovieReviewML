import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer

# RUN nltk.download('stopwords') if you don't have already stopwords installed

def pos_files(files):
    corpus = []
    i=1
    for file in files:
        corpus = treat_data(read_file(file))
        print("Arquivo " + str(i) + " lido!")
        i += 1

def import_data(file):
    # saves file review to string
    dataset = open(file)
    dataset = dataset.read()
    return dataset

def treat_data(data):
    # replace non-alphabetical characters with space
    new_data = re.sub('[^a-zA-Z]', ' ', data)
    new_data = data
    # all to lower case
    new_data = new_data.lower()
    # split words
    new_data = new_data.split()

    ps = PorterStemmer()
    # remove morphological affixes from words
    new_data = [ps.stem(word) for word in new_data if not word in set(stopwords.words('english'))]
    new_data = ' '.join(new_data)

    return new_data

def vectorize(corpus, reviews):
    #interminado
        cv = CountVectorizer(max_features=None)
        X = cv.fit_transform(corpus).toarray()

        print("X size: " + str(len(X)))

def main():
    file = "pos/0_10.txt"
    corpus = []

    data = import_data(file)

    corpus.append(treat_data(data))

    print(corpus)

    vectorize(corpus, data)

main()
