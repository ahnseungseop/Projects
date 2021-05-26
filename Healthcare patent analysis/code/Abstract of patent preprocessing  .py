# Preprocessing 

#%%
# Download library

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#%%

data = pd.read_csv("C:/User/)

data.head()

#%%

# delete and/or , e.g : 특허에 많이 들어있는 표현 불용어 사전에 있지 않아, 따로 삭제 진행

for i in range(0,len(data)):
    data['abstract'][i]=data['abstract'][i].replace('and/or','')
    
for j in range(0,len(data)):
    data['abstract'][j]=data['abstract'][j].replace('e.g.','')
    

content = data[['abstract']]

print(len(content))

print(content.head())

content['abstract'][0]

#%%

# 문자만 추출

for i in range(0, len(content)):
    content['abstract'][i]=re.sub('[^a-zA-Z]',' ', content['abstract'][i])

print(content['abstract'][0])

#%%

# 숫자제거

no_num=re.compile('[^0-9]')

for i in range(0, len(content)):
    content['abstract'][i]="".join(no_num.findall(content['abstract'][i]))
    
print(content['abstract'][0])

#%%

# 특수문자 제거

for i in range(0, len(content)):
    content['abstract'][i]=re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','',content['abstract'][i])
    
print(content['abstract'][0])

#%%

# 소문자화

for i in range(0, len(content)):
    content['abstract'][i]=content['abstract'][i].lower()
    
print(content['abstract'][0])

#%%

# 토큰화

content['abstract'] = content.apply(lambda row : nltk.word_tokenize(str(row['abstract'])),axis=1)

print(content.head(5)) # 상위 5개만 뽑아 단어 토큰화 결과 확인 

#%%

# delete nltk stopwords

stop = stopwords.words('english')
content['abstract'] = content['abstract'].apply(lambda x: [word for word in x if word not in (stop)])
print(content.head(5)) # a, and, for와 같은 불용어 사라짐


#%%

# lemmatizer

# 표제어 추출, 3인칭 단수표현을 1인칭으로 과거 동사를 현재형으로 변환 등과 같은 작업 수행
content['abstract'] = content['abstract'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
print(content.head(5))

print(content['abstract'][0])


#%%

# delete too short words

# 길이가 3개 이하인 단어는 제거

tokenized_doc =content['abstract'].apply(lambda x: [word for word in x if len(word) > 3])
print(tokenized_doc[:5])

print(tokenized_doc[0])


#%%

# Combine token

for i in range(0, len(tokenized_doc)):
    tokenized_doc[i] = " ".join(tokenized_doc[i])
    
print(tokenized_doc[0])

#%%

# write excel

tokenized_doc.to_excel('tokenized_result.xlsx')
