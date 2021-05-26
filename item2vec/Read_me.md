# Item2Vec

### 연구목적
<br>

- Word2Vec 기법을 활용한 Item2Vec의 아이디어와 튜토리얼 제공.<br>
<br>

- Word2Vec은 단어의 의미적 특성과 단어간의 관계를 보존하는 단어 임베딩 벡터 생성 <br>
<br>

- Item2Vec은 이러한 Word2Vec의 장점을 살려서 Item에 관한 각 소비자의 선호 관계를 보전하는 Item 임베딩 벡터 생성
<br>

- 기존의 협업필터링 SVD 방식과 비교하여, 임베딩 벡터의 성능이 우수하다고 알려져 있음
<br>

- 본 코드는 영화 시청 및 평점 데이터를 활용하여, Item2Vec을 구현 
<br>


### 실행 순서

#### 1. 데이터 다운로드
    - movies(https://drive.google.com/file/d/1lXZ8dMJoOJ3OjiOokhZCLOxMDW0rR132/view?usp=sharing)
    - ratings(https://drive.google.com/file/d/1SJTfnejCA46O001Y6UjyzPZ25vT4oyoU/view?usp=sharing)

#### 2.code>item2vec.ipynb파일을 이용하여 Item2Vec Tutorial 수행<br>  

### 활용 시나리오

![image.png](attachment:image.png)



![image-2.png](attachment:image-2.png)

- UP과 비슷한 임베딩 벡터를 가진 영화 목록 : 토이스토리3, 라따뚜이 등
- UP을 시청한 소비자에게 비슷한 임베딩 벡터를 가진 영화들을 추천 가능
- 실제로 장르가 비슷한 영화들이 추천됨

![image.png](attachment:image.png)

![image-2.png](attachment:image-2.png)


- Matrix와 Django의 임베딩 벡터를 연산하면, 배트맨 다크나이트의 임베딩 벡터가 나옴
- 이를 통해 Matrix와 Django에 높은 평점을 준 소비자에게 배트맨 다크나이트를 추천해줄 수 있음


```python

```
