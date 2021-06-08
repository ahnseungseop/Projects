# 필요패키지 다운로드

import cv2
import dlib
import numpy as np
import os
import imutils

#%%

# set directories
os.chdir('./data')
mode ='.png'

# 마스킹 할 사진 지정
path1 = 'maksssksksss841'
path = mode+path1

#Initialize color [color_type] = (Blue, Green, Red)
color_blue = (255,255,255)
color_cyan = (255,200,0)
color_white = (0, 0, 0)

#%%
# set directories
os.chdir('./data')
mode ='.png'

# 마스킹 할 사진 지정
path1 = 'maksssksksss841'
path = mode+path1

#Initialize color [color_type] = (Blue, Green, Red)
color_blue = (255,255,255)
color_cyan = (255,200,0)
color_white = (0, 0, 0)

#%%
# Loading the image and resizing, converting it to grayscale

img= cv2.imread('image_name')

gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#%%
# 얼굴 탐색
detector = dlib.get_frontal_face_detector()

faces = detector(gray, 1)

# printing the coordinates of the bounding rectangles
print(faces)
print("Number of faces detected: ", len(faces))

#%%

# 학습된 모델 불러오기
p = "C:/Users/inolab/Desktop/DNN/CNN_homework/faces/shape_predictor_68_face_landmarks.dat"

# Initialize dlib's shape predictor
predictor = dlib.shape_predictor(p)

# 여러 얼굴이 등장하는 사진일 때 특정 얼굴 index로 선택

landmarks=predictor(gray, faces[0])

# 사진 내 여러 얼굴의 좌표 설정

points = []
for i in range(1, 16):
    point = [landmarks.part(i).x, landmarks.part(i).y]
    points.append(point)
print(points)

#%%

# 원하는 파라미터 설정
# Use input () function to capture from user requirements for mask type and mask colour
choice1 = input("Please select the choice of mask color\nEnter 1 for blue\nEnter 2 for white:\n")
choice1 = int(choice1)

if choice1 == 1:
    choice1 = color_blue
    print('You selected mask color = blue')
elif choice1 == 2:
    choice1 = color_white
    print('You selected mask color = white')
else:
    print("invalid selection, please select again.")
    input("Please select the choice of mask color\nEnter 1 for blue\nEnter 2 for white :\n")


choice2 = input("Please enter choice of mask type coverage \nEnter 1 for correct \nEnter 2 for chin \nEnter 3 for nose :\n")
choice2 = int(choice2)

if choice2 == 1:
    
    print(f'You chosen correct mask') # 올바른 마스크
elif choice2 == 2:
    
    print(f'You chosen chin incorrect mask') # 턱스크
elif choice2 == 3:
    
    print(f'You chosen  nose incorrect mask') # 코스트
else:
    print("invalid selection, please select again.")
    input("Please enter choice of mask type coverage \nEnter 1 for high \nEnter 2 for medium \nEnter 3 for low :\n")

#%%

# Draw
points = []
for i in range(1, 16):
    point = [landmarks.part(i).x, landmarks.part(i).y]
    points.append(point)
    

   # 올바른 마스크 좌표
mask_a = [((landmarks.part(42).x), (landmarks.part(15).y)),
              ((landmarks.part(27).x), (landmarks.part(27).y)),
              ((landmarks.part(39).x), (landmarks.part(1).y))]

    # 턱스크 좌표
mask_c = [((landmarks.part(54).x), (landmarks.part(11).y)),
              ((landmarks.part(57).x), (landmarks.part(7).y)),
         ((landmarks.part(48).x), (landmarks.part(5).y))]

    # 코스크 좌표
mask_e = [((landmarks.part(35).x), (landmarks.part(35).y)),
              ((landmarks.part(34).x), (landmarks.part(34).y)),
              ((landmarks.part(33).x), (landmarks.part(33).y)),
              ((landmarks.part(32).x), (landmarks.part(32).y)),
              ((landmarks.part(31).x), (landmarks.part(31).y))]

fmask_a = points + mask_a
fmask_c = points + mask_c
fmask_e = points + mask_e


fmask_a = np.array(fmask_a, dtype=np.int32)
fmask_c = np.array(fmask_c, dtype=np.int32)
fmask_e = np.array(fmask_e, dtype=np.int32)

mask_type = {1: fmask_a, 2: fmask_c, 3: fmask_e}
mask_type[choice2]


# change parameter [mask_type] and color_type for various combination
img2 = cv2.polylines(img, [mask_type[choice2]], True, choice1, thickness=2, lineType=cv2.LINE_8)

# Using Python OpenCV – cv2.fillPoly() method to fill mask
# change parameter [mask_type] and color_type for various combination
img3 = cv2.fillPoly(img2, [mask_type[choice2]], choice1, lineType=cv2.LINE_AA)
    
#%%

# 잘 씌워졌는지 확인

cv2.imshow("image with mask", img3)
cv2.waitKey(7777)
cv2.destroyAllWindows()

#%%

#Save the output file for testing

outputNameofImage = path1+'_low.png'
outputNameofImage1 = path1+'_mid.png'

print("Saving output image to", outputNameofImage1)
cv2.imwrite(outputNameofImage1, img3)

