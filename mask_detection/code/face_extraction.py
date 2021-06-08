# 필요 패키지 다운로드
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rn
import tensorflow as tf
import cv2
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET # xml 읽어들이는 코드
import imutils
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

#%%

# 폴더 내 파일 목록 가져오기

import os
Dirname = []
Filenames = []

for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        Dirname.append(dirname)
        Filenames.append(filename)
        X=os.path.join(dirname,filename)
        print(X)
        
#%%

# 이미지 파일과 xml 파일의 개수가 맞는지 확인

Dir = './data'
print(os.listdir(Dir))

images_path = './data/images'
print("Image path = {}".format(images_path))
print("Total number of images : {}".format(len(os.listdir(images_path))))

Annotation_path = './data/annotations'
print('Annotation path = {}'.format(Annotation_path))
print('Total Annotation files are {}'.format(len(os.listdir(Annotation_path))))

#%%

# 이미지 파일과 xml 파일이 쌍으로 잘 이루어졌는지 확인

Image_width = 80
Image_height = 80
Image_array = []
Labels = []

#Check label files are according to images files
Sorted_files = sorted(os.listdir(Annotation_path))

Sorted_images_path = sorted(os.listdir(images_path))


for i in range(0,len(Sorted_files)):
    print(Sorted_files[i],Sorted_images_path[i])
    
#%%

# xml 파일로 얼굴 추출 함수 정의

def get_box(obj):
    
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]


#%%

# 각 이미지에서 xml에 있는 얼굴 좌표를 이용해서 얼굴 이미지 추출하기

for file in tqdm(sorted(os.listdir(Annotation_path)),desc='Preparing data..'):
    file_path = Annotation_path + "/" + file
    xml = ET.parse(file_path)
    root = xml.getroot()
    image_path = images_path + "/" + root[1].text

     
    for bndbox in root.iter('bndbox'):
        [xmin, ymin, xmax, ymax] = get_box(bndbox)
        img = cv2.imread(image_path)
        face_img = img[ymin:ymax,xmin:xmax]
        face_img  = cv2.resize(face_img,(Image_width,Image_height))
        Image_array.append(np.array(face_img)) 
    
    for obj in root.findall('object'):
        name = obj.find('name').text 
        Labels.append(np.array(name)) 
        
#%%

# Nomalize the data
num_classes = 3
X = np.array(Image_array)
X = X/255

le = LabelEncoder()
y = le.fit_transform(Labels)
y  = to_categorical(y,num_classes)
print(y)

print(X.shape)
print(y.shape)

#%%

# check random image
fig, ax = plt.subplots(2,2)
fig. set_size_inches(10,10)

for i in range(2):
    for j in range(2):
        l = rn.randint(0, len(Labels))
        
        image = cv2.cvtColor(Image_array[l], cv2.COLOR_BGR2RGB)
        
        ax[i,j].imshow(image)
        ax[i,j].set_title(Labels[l])

#%%

# 이미지 내보내기

for i in tqdm(range(0,len(Image_array))):
    image = cv2.cvtColor(Image_array[i], cv2.COLOR_BGR2RGB)
    cv2.imwrite('C:/Users/inolab/Desktop/DNN/mask_detection/sample'+str(i)+'.png', image)

#%%

