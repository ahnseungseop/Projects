# Healthcare patent analysis

### 연구목적
<br>

- 헬스케어의 패러다임이 개인의 건강을 스스로 관리하고 예방하는 추세로 변화.<br>
<br>

- 헬스케어 산업과 ICT 기술이 융합된 '디지털 헬스케어'가 등장 <br>
<br>

- 기업이 시장에서 경쟁우위를 차지하기 위해서는 R&D에 앞서 산업의 기술 구조 및 핵심 기술에 대한 파악이 필요
<br>

- 기술의 구조 분석에 이용되었던 토픽모델링 기법과 함께 각종 문서 분류에 활용되고 있는 Doc2Vec 기법을 이용한 분석을 통해 보다 유의믜한 토픽을 추출 및 비교해 볼 수 있음.
<br>

- 텍스트마이닝 기법을 활용하여 디지털 헬스케어 기술의 구조 분석
<br>

- 기술 구조 분석의 기법으로서 토픽모델링(fractioinal assigment)과 Doc2Vec 기반 클러스터링(discrete assigment) 특성 비교
<br>

- 본 방법론은 디지털 헬스케어 뿐만, 아니라 다양한 분야의 특허 데이터에도 적용 가능함.


### 실행 순서

#### 1. Abstract patent preprocessing.py로 특허 데이터의 abstarct 부분의 전처리 진행<br>
    - tokenized_result.xlsx이 결과 파일로 저장됨


#### 2. Patent_Topic modeling_analysis.R을 이용해서 tokenized_result.xlsx로 Topic modeling 수행.<br>   
    - topic-term_list_patent_t.csv 및 probabaility_topic_term_patent_t.csv 파일 저장됨<br>

    
#### 3. Topic modeling_tidyup_result.R을 이용하여 각기 다른 결과파일을 하나의 excel 파일로 묶어줌
<br>

#### 4. (1)에서 출력된, tokenized_result.xlsx 파일을 이용하여 Doc2Vec and K-means_clustering.py 파일 실행
    - Doc2Vec을 수행한 벡터값 Doc2Vec_vector.xlsx 저장됨
    - Doc2Vec_vector.xlsx를 이용하여, k-means clustering을 수행한 vec_df_kmeans.xlsx 파일 저장됨



```python

```
