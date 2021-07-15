
import re
from nltk.corpus import stopwords
import nltk
from bs4 import BeautifulSoup


# 불용어 업로드
#nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# 어간 추출을 위한 어간 추출 알고리즘
stemmer = nltk.stem.SnowballStemmer('english')

# 영어 전처리 메소드
def data_text_cleaning(text):

    # email 제거
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
    data_text = re.sub(pattern=pattern, repl=' ', string=text)

    # url 제거
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    data_text = re.sub(pattern=pattern, repl=' ', string=data_text)

    # 해쉬태그 부분 제거
    pattern = '(@[a-zA-Z0-9-]+(\s|:))'
    data_text = re.sub(pattern=pattern, repl=' ', string=data_text)

    # RT 제거
    pattern = '(RT+(\s|:))'
    data_text = re.sub(pattern=pattern, repl=' ', string=data_text)

    # MKR 제거
    pattern = '(#mkr)'
    data_text = re.sub(pattern=pattern, repl=' ', string=data_text)

    # HTML 변환
    data_text = BeautifulSoup(data_text, 'html.parser').get_text()

    # 영어인 문자만 남기도록
    data_text = re.sub('[^a-zA-Z]', ' ', data_text)

    # 대문자를 전부 소문자로 치환
    data_text = data_text.lower().split()

    # 불용어 제거 처리
    data_text = [word for word in data_text if not word in stop_words]

    # 어간 추출 : 어형이 과거형이든 미래형이든 하나의 단어로 취급하기 위한 처리작업
    data_text = [stemmer.stem(word) for word in data_text]

    return ' '.join(data_text)