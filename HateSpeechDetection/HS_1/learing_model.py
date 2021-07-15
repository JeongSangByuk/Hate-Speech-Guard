import pickle

from imblearn.over_sampling import SMOTE
import numpy as np
from gensim.models import FastText
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical

#np.load 가 보안의 문제로 막혀있을 경우, allow_pickle=True 로 설정
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
data = np.load('../HateSpeechDataSet/csv3/hs_total_data_cleanded.npy')
np.load = np_load_old

X = []
Y = []
categroy_class = ['none','obscene','offensive','racism','sexism']
number_category = len(categroy_class)
max = 100
output_dim = 300

for i in range(len(data)):
    X.append(data[i][0])
    Y.append(data[i][1])

#라벨 인코딩 //encoder.classes_로 확인가능!
# transform 함수는 카테고리를 정수 인덱스 리스트로 변환
encoder = LabelEncoder()
encoder.fit(categroy_class)
Y = encoder.transform(Y)

#pickle을 활용한 token 호출
with open('../HateSpeechDataSet/csv3/hs_total_data_cleanded.pickle', 'rb') as handle:
    token = pickle.load(handle)

Xtoken = token.texts_to_sequences(X)
Xpad = pad_sequences(Xtoken, max)

# # 데이터 불균형 해소를 위한 SMOTE 적용
X_resampled, Y_resampled = SMOTE(random_state=0).fit_resample(Xpad, Y)

# 비율 그래프 출력 메소드
# def count_and_plot(y):
#     counter = Counter(y)
#     pyplot.bar(counter.keys(), counter.values())
#     pyplot.show()
#count_and_plot(Y_resampled)

#Embedding의 첫번째 인자값, 텍스트 데이터의 전체 단어 집합의 크기+1
wordsize = len(token.word_index) + 1

#정수 인코딩 된 결과로부터 원-핫 인코딩을 수행하는 to_categorical 메소드
YoneHot = to_categorical(Y_resampled)

#train_test_split 함수는 전체 데이터셋 배열을 받아서 랜덤하게 훈련/테스트 데이터 셋으로 분리해주는 함수
Xtrain,Xval,Ytrain,Yval = train_test_split(X_resampled,YoneHot, test_size=0.2)

# ###################fasttext 사용########################
# 구글에서 제작한 pretrained model (워드투백모델)
# model_fasttext = KeyedVectors.load_word2vec_format('HateSpeechDataSet/FastText/GoogleNews-vectors-negative300.bin.gz', binary=True)
# 내가 제작한 pretrained model
model_fasttext = FastText.load('HateSpeechDataSet/FastText/fasttext.model')

def get_vector(word):
    if word in model_fasttext:
        return model_fasttext[word]
    else:
        return None

embedding_vocab = np.zeros((wordsize,output_dim))

for word,index in token.word_index.items():
    # 단어(key) 해당되는 임베딩 벡터의 input_dim 의 값(value)를 임시 변수에 저장
    temp = get_vector(word)
    # 만약 none이 아니라면 벡터의 값을 리턴받은 것이므로 저장
    if temp is not None:
        embedding_vocab[index] = temp
# #########################################################


model = Sequential()

# 첫번째 인자 : 텍스트 데이터의 전체 단어 집합의 크기, 두번째 인자 : 임베딩 되고 난 후의 단어의 차원, 세번째 인자 : 입력 시퀀스의 길이
# Embedding() 단어를 밀집 벡터로 만드는 역할->임베딩 층을 만든다.
# word2Vec 사용시 임베딩
model.add(Embedding(input_dim=wordsize,output_dim=output_dim,input_length=max,weights=[embedding_vocab]))
# 그냥 임베딩
#model.add(Embedding(input_dim=wordsize, output_dim=output_dim, input_length=max))

############################### 모델 설정 1 : LSTM #########
# model.add(LSTM(60,return_sequences=True))
# model.add(GlobalMaxPool1D())
# model.add(Dropout(0.2))
# model.add(Dense(50,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(number_category, activation='softmax'))
################################################################

############################### 모델 설정 2 : BiLSTM #########
# model.add(Bidirectional(LSTM(50,return_sequences=True,recurrent_dropout=0.15)))
# model.add(GlobalMaxPool1D())
# model.add(Dropout(0.2))
# model.add(Dense(50,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(number_category, activation='softmax'))
################################################################

############################### 모델 설정 3 : 1D CNN #########
model.add(Dropout(0.2))
model.add(Conv1D(32, 5, strides=1, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(number_category, activation='sigmoid'))
################################################################

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

hist = model.fit(Xtrain,Ytrain,batch_size=128,epochs=10,validation_data=(Xval,Yval),validation_split=0.2)

print("정확도 : %.4f" % (model.evaluate(Xtrain, Ytrain)[1]))

#model.save('model_cnn.h5')