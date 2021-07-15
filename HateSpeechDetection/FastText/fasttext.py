import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from gensim.models import FastText

#np.load 가 보안의 문제로 막혀있을 경우, allow_pickle=True 로 설정
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
data = np.load('../HateSpeechDataSet/csv3/hs_total_data_cleanded.npy')
np.load = np_load_old

X = []

for i in range(len(data)):
    X.append(data[i][0].split())

embedding = FastText(X,size = 300,window=5,min_count=5,negative=3)

# 모델 저장
embedding.save('HateSpeechDataSet/FastText/fasttext.model')

# model = FastText.load('HateSpeechDataSet/FastText/fasttext.model')
# print(model.wv.most_similar('fuck'))