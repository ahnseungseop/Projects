## 오착용 데이터를 추가한 mask-detection

### 연구 배경

- Covid-19 확산으로 인한 마스크 착용의 생활화 <br> 
<br>

- 확산 방지 차원에서 여러 공공장소에서 마스크 착용과 체온을 측정하는 기기 구비<br>  
<br>

- 관리인 비상주 점포가 늘어남에 따라 마스크 오착용을 판별하여 입장 불가를 알려주는 시스템 필요성 증가<br> 

### 연구 목적

 - 마스크 착용을 판별하는 여러 선행연구가 있지만, 착용과 미착용만을 분류하고 있음.<br>
 <br>
 
 - 현실에서는 마스크 오착용하는 경우도 많기 때문에, 이를 판별할 수 있는 모델의 필요성이 있음.

### 데이터

- 원 데이터는 kaggle mask-detection dataset을 활용함.<br>
<br>

- Raw data : https://drive.google.com/file/d/11rmxPFncWOtgheGmenVfe_zNWPTRODVg/view?usp=sharing<br>
<br>

- 거기에 dlib library를 이용하여, 총 150개의 오착용 데이터를 추가함.<br>
<br>

- 추가한 데이터 set에 원 dataset에 제공되는 xml data를 활용하여, 얼굴만을 추출함.<br>
<br>

- dataset url : https://drive.google.com/file/d/1aCYTMsMm18Ocf_ykjj5tp6FVKDfX1Y3E/view?usp=sharing

### 연구방법
- Raw data와 masking.py를 이용하여, 특정 얼굴에 턱스크와 코스크를 씌움으로써 오착용 데이터 생성<br>
<br>

- 생성된 오착용 데이터와 원본 데이터를 합쳐, face_extraction.py를 이용하여 얼굴만을 추출<br>
<br>

- 추출된 얼굴을 ResNet-18을 사용하여 학습(mask_detection_code.py) <br>
<br>

- epoch 40, batch_size 64, optimizer SGD, learning rate 0.01, momentum 0.5, weight decay 0.001<br>
<br>

- 분류 성능을 비교할 baseline model은 https://www.kaggle.com/prekshabhavsar/face-mask-detection-with-92-accuracy 참고 

### 연구 결과

- test_accuracy : 94.5 <br>
<br>
 
- test_loss : 0.2947<br>
<br>

- 마스크 미착용 f-1 score : 91.86 / 마스크 착용 f-1 score : 96.70 / 마스크 오착용 f-1 score : 72.07<br>
<br>
- base_line 모델보다 성능 향상 및 오착용 데이터 분류 성능 향상


```python

```
