from youtube_transcript_api import YouTubeTranscriptApi
import pickle
import re
from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import load_model
import numpy as np
import nltk
from lime.lime_text import LimeTextExplainer
from Utils.Utils import data_text_cleaning
from timeit import default_timer as timer

model = load_model('HS_3/model_cnn_1.h5')

category_class = ['hate','none']
number_category = len(category_class)
max = 50
output_dim = 300
start_time = timer()

#pickle을 활용한 token 호출
with open('HS_3/csv3/hs_total_data_cleanded.pickle', 'rb') as handle:
    token = pickle.load(handle)

def print_now_time(now):
    now_time = timer()
    print(now,'           ', now_time - start_time , ' 경과')
    return

def predict_result(s):
    if not s:
        return 1

    sentence = token.texts_to_sequences(s.split(' '))
    # wow: 412  => wow는 토큰 인덱스 412번째, 토근사전에 없는 단어가 들어갈 경우 의미없는 단어처리(=wow)
    for i in range(len(sentence)):
        if not sentence[i]:
            sentence[i].append(412)

    sent_pad = pad_sequences([sentence], maxlen=max)
    k = model.predict(sent_pad)
    return np.argmax(k[0])

def clean_caption_list_string(list_string) :
    if list_string == 'None':
        return list_string
    return list_string[3:5]

# 수동생성인 자막 언어 리스트 추출
def get_manually_caption_list(video_id) :
    lan = str(YouTubeTranscriptApi.list_transcripts(video_id)).split('(MANUALLY CREATED)')[1]
    lan = str(lan).split('(GENERATED')[0]
    list = []
    for i in range(len(lan.split('\n'))):
        if not lan.split('\n')[i]:
            continue
        list.append(clean_caption_list_string(lan.split('\n')[i]))

    return list

# 자동생성인 자막 언어 리스트 추출
def get_auto_generated_caption_list(video_id) :
    lan = str(YouTubeTranscriptApi.list_transcripts(video_id)).split('(GENERATED)')[1]
    lan = str(lan).split('(TRANSLATION LANGUAGES)')[0]
    list = []
    for i in range(len(lan.split('\n'))):
        if not lan.split('\n')[i]:
            continue
        list.append(clean_caption_list_string(lan.split('\n')[i]))
    return list

def get_caption(url) :
    # videoId = url의 (v=) 다음에 오는 11개 문자열 compile
    pat = re.compile("(v=)([a-zA-Z0-9-_]{11})")
    # v= 문자열을 검색할땐 포함하지만 search할 땐 그룹으로 제외해준다.
    video_id = pat.search(url).group(2)

    # 생성된 자막이 아예없을 경우를 위한 예외처리 ->
    try:
        list_manually_generated_caption = get_manually_caption_list(video_id)
        list_auto_generated_caption = get_auto_generated_caption_list(video_id)
        print_now_time('complete getting caption')
    except:
        print("no Caption")
        return

    print_now_time('start predicting')
    # case 1 : en , none   /   case 2 : none , en    /   case 3 : none and none

    # 수동 생성된 자막에 en 이 있는 경우 // red - 31 , blue - 34
    if 'en' in list_manually_generated_caption or 'en' in list_auto_generated_caption:

        data = YouTubeTranscriptApi.get_transcript(video_id, languages = {'en'})
        for i in range(len(data)):

            text = data_text_cleaning(data[i]['text'])
            category_sentence = predict_result(text)

            if(category_sentence == 0) :
                now_time = timer()
                end = float(data[i]['start']) + float(data[i]['duration'])
                print('classified   :   ' + str(data[i]['start']) + " ~ " + str(end) , '           ', now_time - start_time, ' 경과')
            # else:
            #     print('classified   :   ' + str(data[i]['text']))
        print(' ****************** Manually generated ******************')
        print_now_time('finished')

    # 자동 생성된 자막에 en 이 있는 경우
    # elif 'en' in list_auto_generated_caption:
    #     data = YouTubeTranscriptApi.get_transcript(video_id, languages={'en'})
    #     for i in range(len(data)):
    #         print(str(data[i]['text']))
    #     print("\n ****************** Auto generated ******************")

    # 수동,자동 둘다 en이 없는 경우
    else :
        print("no English caption result")

url = 'https://www.youtube.com/watch?v=2VZc0XaTGqM'
url2 = 'https://www.youtube.com/watch?v=2VZc0XaTGqM'
url3 = 'https://www.youtube.com/watch?v=yIvb4csSgcs'
url4 = 'https://www.youtube.com/watch?v=SJOnhWiJArM'

get_caption(url4)