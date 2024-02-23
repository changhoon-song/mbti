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


my_posts = """Hi I am 21 years, currently, I am pursuing my graduate degree 
in computer science and management (Mba Tech CS ), 
It is a 5-year dual degree.... My CGPA to date is 3.8/4.0 . 
I have a passion for teaching since childhood. 
Math has always been the subject of my interest in school. 
Also, my mother has been one of my biggest inspirations for me. 
She started her career as a teacher and now has her own education trust 
with preschools schools in Rural and Urban areas. During 
the period of lockdown, I dwelled in the field of blogging and content creation on Instagram.  
to spread love positivity kindness . r
I hope I am able deliver my best to the platform and my optimistic attitude helps in the growth that is expected. 
Thank you for the opportunity."""

y_pred_result = classifier(my_posts)  
print("MBTI 결과:", y_pred_result)




