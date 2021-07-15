import pickle
import re
import os
import warnings
import nltk

# 모든 경고 무시
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import load_model
import numpy as np
from lime.lime_text import LimeTextExplainer
from nltk import tokenize
from Utils.Utils import data_text_cleaning

model = Sequential()
model = load_model('HS_3/model_lstm_1.h5', compile=True)

category_class = ['hate', 'none']
number_category = len(category_class)
max = 50

# 문장 split을 위한 punkt 다운로드 => 맨처음 다운로드 이후 주석 처리
nltk.download('punkt')

# pickle을 활용한 token 호출
with open('HS_3/csv3/hs_total_data_cleanded.pickle', 'rb') as handle:
    token = pickle.load(handle)

origin_total_text = " “Fuck you, nigger, go back to Africa. The slave ship is loading up,” he said."

# 문장별로 split 된 배열 생성
sentence_splited = tokenize.sent_tokenize(origin_total_text)

result = ""


def predict_result(s):
    sentence = token.texts_to_sequences(s)
    sent_pad = pad_sequences(sentence, max)
    k = model.predict(sent_pad)

    return k


def masking_hate_speech(origin_text, cleaned_text):
    explainer = LimeTextExplainer(class_names=category_class)
    explanation = explainer.explain_instance(cleaned_text, predict_result)

    deleted_word_array = []
    hate_word_array = explanation.as_list()

    for i in range(len(hate_word_array)):
        if (hate_word_array[i][1] < -0.1 and hate_word_array[i][1] < 0):
            deleted_word_array.append(hate_word_array[i])

    masking_text_array = origin_text.split(' ')
    print(masking_text_array)
    print(deleted_word_array)
    for k in range(len(masking_text_array)):

        if not masking_text_array[k]:
            continue

        for kk in range(len(deleted_word_array)):
            if deleted_word_array[kk][0] in masking_text_array[k].lower():
                masking_text_array[k] = "****"

    return ' '.join(masking_text_array)

for i in range(len(sentence_splited)):
    text = data_text_cleaning(sentence_splited[i])
    sentence = token.texts_to_sequences(text.split(' '))

    # wow: 412  => wow는 토큰 인덱스 412번째, 토근사전에 없는 단어가 들어갈 경우 의미없는 단어처리(=wow)
    for j in range(len(sentence)):
        if not sentence[j]:
            sentence[j].append(412)

    sent_pad = pad_sequences([sentence], maxlen=max)
    k = model.predict(sent_pad)

    if category_class[np.argmax(k[0])] == "none":
        result = result + " " + sentence_splited[i]
    else:
        result = result + " " + masking_hate_speech(origin_text=sentence_splited[i], cleaned_text=text)

print(result)