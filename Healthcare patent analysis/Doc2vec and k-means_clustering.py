# Doc2vec and K-means clustering

#%%

import pandas , nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import RegexpTokenizer

#%%

# load data

filename = 'C:/Users/tokenized_result.xlsx'
df=pandas.read_excel(filename)
df.columns=['index','abstract']

df.head()

#%%

# tokenizer 함수 생성

def nltk_tokenizer(_wd):
    return RegexpTokenizer(r'\w+').tokenize(_wd.lower())

df['Token_abstract'] = df['abstract'].apply(nltk_tokenizer)
df.head()

doc_df = df[['index','Token_abstract']].values.tolist()

tagged_data = [TaggedDocument(words=_d, tags=[uid]) for uid, _d in doc_df]

tagged_data [0]

#%%

# doc2vec 

demension = 15

max_epochs = 10

model = Doc2Vec(
    window=10,
    size=demension,
    alpha=0.025, 
    min_alpha=0.025,
    min_count=1,
    dm =1,
    negative = 5,
    seed = 9999)

model.build_vocab(tagged_data)

#%%

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.002
    # 학습률 고정
    model.min_alpha = model.alpha
    

#%%
    
# 0번 문서와 유사한 문서 상위 5개와 코사인 유사도 출력

model.random.seed(9999)
doc=0
dic_sim={}
return_docs = model.docvecs.most_similar(doc,topn=2063)
for rd in return_docs:
    for des in df[df['index'] == rd[0]]['abstract']:
        #print (rd[0],rd[1])
        dic_sim[rd[0]]=rd[1]
        
#%%
        
# similarity matrix (document by document)
        
col=list(range(0,len(df)))

df2 = pandas.DataFrame(index=range(0,len(df)), columns=col)

df2.head()        
        
        
model.random.seed(9999)

for i in range(0,len(df)):
    doc=i
    dic_sim={}
    return_docs = model.docvecs.most_similar(doc,topn=2063)
    for rd in return_docs:
        for dex in df[df['index']==rd[0]]['abstract']:
            dic_sim[rd[0]]=rd[1]
            dic_sim[doc]=0
            a=sorted(dic_sim.items())
    df2.iloc[i]=a
    print(i)
    
#%%
    
for k in range(0, len(df2)):
    for j in range(0, len(df2.columns)):
        df2.iloc[k,j]=df2.iloc[k,j][1]
    print(k)

df2.head()

#%%
    
# doc2vec vector matrix (document by value of demension )

col=list(range(0,demension))

df3 = pandas.DataFrame(index=range(0,len(df)), columns=col)

df3.head()

for i in range(0,len(df)):
    df3.iloc[i]=model.docvecs[i]
    print(i)
    
df3.head()

df3.to_excel('doc2vec_vector.xlsx')

#%%

# k-means

# load library

import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import seaborn as sns

#%%

# load vector data set

vec_df = pd.read_excel('C:/Users/doc2vec_vector.xlsx')
                 
vec_df.drop(vec_df.columns[0], axis=1, inplace=True)
    
vec_df.head()

#%%

# fitting k-means 

cluster=15

data_points = vec_df.values

kmeans_15 = KMeans(n_clusters=cluster).fit(data_points)

#%%

print(kmeans_15.labels_) # document 들의 cluster label. 입력 데이터 순대로 나열

print(kmeans_15.n_iter_) # centroid가 몇번 이동했는가.

#%%

vec_df_kmeans15 = vec_df

vec_df_kmeans15['cluster_id'] = kmeans_15.labels_ 

vec_df_kmeans15.head()

#%%

vec_df_kmeans15.to_excel('vec_df_kmeans15.xlsx')

#%%




