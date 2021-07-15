import pickle
import re
import os
import nltk
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.python.keras.models import load_model
import numpy as np
from bs4 import BeautifulSoup
from lime.lime_text import LimeTextExplainer

from Utils.Utils import data_text_cleaning

model = load_model('model_bilstm_1.h5')

category_class = ['hate','none']
number_category = len(category_class)
max = 50
output_dim = 300

#pickle을 활용한 token 호출
with open('csv3/hs_total_data_cleanded.pickle', 'rb') as handle:
    token = pickle.load(handle)

text = data_text_cleaning("“fuck you, nigger, go back to Africa. The slave ship is loading up,” he said.")
print(data_text_cleaning(text))

sentence = token.texts_to_sequences(text.split(' '))
# wow: 412  => wow는 토큰 인덱스 412번째, 토근사전에 없는 단어가 들어갈 경우 의미없는 단어처리(=wow)
for i in range(len(sentence)):
    if not sentence[i]:
        sentence[i].append(412)

sent_pad = pad_sequences([sentence], max)
k = model.predict(sent_pad)

print("각 클래스마다 확률 : ", k[0])
print("예측 클래스 : ", category_class[np.argmax(k[0])])


def predict_result(s):
    sentence = token.texts_to_sequences(s)
    sent_pad = pad_sequences(sentence, max)
    k = model.predict(sent_pad)

    print("Lime 각 클래스마다 확률 : " ,k[0])
    print("Lime 예측 클래스 : ", category_class[np.argmax(k[0])])
    return k

# Lime 로컬대리분석 기법 이용.
explainer = LimeTextExplainer(class_names=category_class)
explanation = explainer.explain_instance(text, predict_result)

# 평가파일 생성.
num = "1"
file_name = "predict_" + num
while(1):
    if(os.path.isfile(file_name+".html")):
        file_name = file_name + num
        continue
    else:
        print("file name : ",file_name)
        explanation.save_to_file(file_name+".html")
        break

#explanation.show_in_notebook(text=True)
#print(explanation.as_list())
#explanation.as_pyplot_figure()