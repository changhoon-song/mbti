import pandas as pd
import numpy as np
# 데이터 불러오기
data = pd.read_csv(r"D:\bigdata_itwill\semi_project\data\MBTI500.csv")
df = data.copy()

# 데이터 정보 확인
df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 106067 entries, 0 to 106066
Data columns (total 2 columns):
 #   Column  Non-Null Count   Dtype 
---  ------  --------------   ----- 
 0   posts   106067 non-null  object
 1   type    106067 non-null  object
dtypes: object(2)
memory usage: 1.6+ MB
"""

# 데이터에 null값 확인
df.isnull().any()
"""
posts    False
type     False
dtype: bool
"""

# 데이터 type의 종류
np.unique(np.array(df['type']))
"""
array(['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
       'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'],
      dtype=object)
"""

# 타입별 게시글 수
post_type = data.groupby(['type']).count()
post_type


# 텍스트 데이터 수치로 변경
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# 피쳐화 진행 시 메모리가 너무 큰 관계로 max_features = 8000 설정
cvect = CountVectorizer(max_features=8000)
dtm = cvect.fit_transform(df["posts"])
DTM_array = dtm.toarray()

# 타겟 전처리
label_encoder = LabelEncoder()
target = df['type']
target = label_encoder.fit_transform(target)

label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
label_mapping
"""
{0: 'ENFJ',
 1: 'ENFP',
 2: 'ENTJ',
 3: 'ENTP',
 4: 'ESFJ',
 5: 'ESFP',
 6: 'ESTJ',
 7: 'ESTP',
 8: 'INFJ',
 9: 'INFP',
 10: 'INTJ',
 11: 'INTP',
 12: 'ISFJ',
 13: 'ISFP',
 14: 'ISTJ',
 15: 'ISTP'}
"""

# train/test split : 훈련셋(70) vs 테스트셋(30)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    DTM_array, target, test_size=0.3)

print((x_train.shape), (y_train.shape), (x_test.shape), (y_test.shape))
# (74246, 8000) (74246,) (31821, 8000) (31821,)

# Naive Bayes 분류기
from sklearn.naive_bayes import MultinomialNB # nb model
from sklearn.metrics import accuracy_score 

# 학습 모델 만들기 : 훈련셋 이용
nb = MultinomialNB()
model = nb.fit(X= x_train, y = y_train)

# 학습 model 평가 : 테스트셋 이용
y_pred = model.predict(X = x_test)

# 분류정확도 
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('분류정확도 :', acc) # 분류정확도 : 0.7334778919581408


# 문서분류기 함수
def classifier(texts):
    global model, cvect, label_mapping
    
    DTM_test = cvect.transform([texts])
    X_test = DTM_test.toarray()
    
    y_pred = model.predict(X=X_test)
    y_pred_result = label_mapping[y_pred[0]]
    
    return y_pred_result


my_posts = """I am deeply driven by a purpose that goes beyond the surface, 
seeking to make a meaningful impact in the world through my work and personal interactions. 
My approach to life is characterized by a deep empathy for others, an intuitive understanding of 
human dynamics, and a strong desire to contribute to the greater good. My vision is to be part of 
initiatives that not only challenge me intellectually but also align with my core values of integrity, compassion, 
and innovation.Throughout my experiences, I have honed my ability to listen attentively and 
communicate effectively, skills that have allowed me to connect with individuals 
from diverse backgrounds and perspectives. I am particularly drawn to roles that 
enable me to utilize my insights into human behavior to foster understanding and positive change. 
My analytical skills, coupled with a creative mindset, allow me to approach problems from unique 
angles, finding solutions that are both effective and ethically sound.As an advocate for lifelong 
learning, I continuously seek out opportunities to grow both personally and professionally. 
I am deeply reflective, constantly evaluating my experiences to glean lessons that can inform 
future decisions. This introspective nature has equipped me with a profound understanding of 
my strengths and areas for growth, enabling me to navigate challenges with resilience and grace.
In collaborative environments, I strive to create a sense of harmony and inclusivity, 
believing that the best outcomes are achieved when every voice is heard and valued. 
My leadership style is characterized by a quiet confidence and a focus on empowering others, 
fostering a culture of mutual respect and shared purpose.I am eager to contribute to an organization 
that values innovation, ethical responsibility, and a commitment to making a positive 
difference in the world. My goal is to apply my unique blend of empathy, strategic thinking, 
and dedication to meaningful causes, helping to drive forward initiatives 
that have a lasting impact on communities and society at large."""

y_pred_result = classifier(my_posts)  
print("MBTI 결과:", y_pred_result)




