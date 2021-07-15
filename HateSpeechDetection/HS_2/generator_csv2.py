import csv
import pickle
from collections import Counter

import pandas as pd
import re
import nltk
import math

from keras_preprocessing.text import Tokenizer
from matplotlib import pyplot
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import numpy as np

data = pd.read_csv('../HateSpeechDataSet/csv2/annotations_metadata.csv3')

X = data['file_id']
Y = data['label']
X_text = []

for i in range(len(X)):
    f = open("HateSpeechDataSet/csv2/all_files/" + X[i] + ".txt", 'r', encoding='UTF-8')
    text = f.read()
    X_text.append(text)
    f.close()

f = open('../HateSpeechDataSet/csv2/hs_data.csv3', 'w', newline='', encoding='UTF-8')
wr = csv.writer(f)
for i in range(len(X)):
    wr.writerow([X_text[i],Y[i]])
f.close()


