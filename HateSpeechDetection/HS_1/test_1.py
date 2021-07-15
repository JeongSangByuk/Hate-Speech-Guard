import pickle
import re

import nltk
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.python.keras.models import load_model
import numpy as np
from bs4 import BeautifulSoup

from Utils.Utils import data_text_cleaning

model = load_model('model_cnn.h5')

category_class = ['none','obscene','offensive','racism','sexism']
number_category = len(category_class)
max = 100

#pickle을 활용한 token 호출
with open('../HateSpeechDataSet/csv3/hs_total_data_cleanded.pickle', 'rb') as handle:
    token = pickle.load(handle)

text = data_text_cleaning("@ummayman90 Yes, might, power, and imperialism is what it is all about.  There is zero spirituality in it.  Soon the world will understand.")
print(text)

sentence = token.texts_to_sequences(text)
sent_pad = pad_sequences(sentence, max)
y_predict = model.predict(sent_pad)

arr = np.array(y_predict)
result_print = ""

for number_category in range(number_category):
    result_print =result_print + category_class[number_category] + " : " + "%0.5f" %(arr[0][number_category]*100) + " , "

print(result_print)



print(category_class[np.argmax(y_predict[0])]) #
