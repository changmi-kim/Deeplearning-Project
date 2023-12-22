# Human-Robot Interaction System with ROS2
- (대표 이미지.gif)

## 시스템 구성
- (시스템 구성도 이미지.jpg)

## 팀원 소개 및 역할
|구분|이름|역할|
|---|---|---|
|팀장|한승준|전체 시스템 구성도 제작, Hands landmak 머신러닝 모델 제작 및 성능분석, 주행로봇 제작, 모바일 앱 제작|
|팀원|김창미|Hands landmak 딥러닝 모델 제작 및 성능분석|
|팀원|박한규|Semantic segementation 모델(yolov5 & yolov8) 제작 및 성능분석|

## 프로젝트 기간
2023.11.25 ~ 2023.12.14

## 기술 스택
### 개발환경
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=white)
![Arduino](https://img.shields.io/badge/arduino-00878F?style=for-the-badge&logo=arduino&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white)
![RDS](https://img.shields.io/badge/AWS%20RDS-527FFF?style=for-the-badge&logo=Amazon%20RDS&logoColor=white)
![Qt](https://img.shields.io/badge/Qt-41CD52?style=for-the-badge&logo=Qt&logoColor=white)
![mitapp](https://github.com/addinedu-ros-3rd/iot-repo-2/assets/81555330/11db2c8f-f4ae-46c0-ae71-d83b6e9e1d5c)
</div>

### 언어
![C++](https://img.shields.io/badge/c++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### 커뮤니케이션
![Slack](https://img.shields.io/badge/slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)

## 프로젝트 소개
### Human-Robot Interaction System with ROS2

1. 사람의 제스처(손, 몸)를 인식하고 주행로봇에게 전달한다.
2. 라인 인식 및 장애물을 인식하고 직진, 좌회전, 우회전 여부를 주행로봇에게 전달한다.
3. ROS2 gazebo simulation을 통해 1번과 2번을 검증을 하고 실제 주행로봇에게 전달한다.

## 프로젝트 목표
1. 머신러닝과 딥러닝 모델 직접 만들기
   - 직접 데이터셋 수집
   - 머신러닝, 딥러닝 모델 학습
   - 모델 간의 성능 비교 및 평가
    
2. ROS2 네트워크 통신이해
   - 노드 생성 및 토픽 발행
   - 구독 방법

## Hand landmarks detection
### Mediapipe
머신러닝
- KNN 모델
  
딥러닝
- LSTM 모델
- RNN 모델

## Semantic segmentation 
Yolov5
- contents

Yolov8
- contents

## Enviroment

Hand landmark detect system
```
dependencies:
  - pip=20.2.2  
  - python=3.7.7
  - numpy=1.19.1
  - pandas=1.1.4
  - plyfile=0.7.2
  - pyyaml=5.3.1
  - tqdm=4.54.1
  - matplotlib=3.3.3 
  - pip:
    - opencv-python==4.2.0.34
    - mediapipe==0.8.4.2
    - addict==2.4.0
    - sklearn==0.0
```

Line detect system
```
```

Minecraft simulation
```
```

Gazebo simulation
```
```

## How to run?
- Download
```
$ git clone https://github.com/addinedu-ros-3rd/deeplearning-repo-1.git
```

- Hands detect system
```
$ cd hands_detect_system
```

- Line detect system
```
$ cd line_detect_system
```

- Minecraft simulation
```
$ cd minecraft_simulation

$ pip install ursina

$ python3 menu.py 
```

- gazebo simulation
```
$ cd gazebo_simulation

$ colcon build
```

## Demo video
<p align=center>
  <a href="https://youtu.be/fBUlsuLVDTE?si=vuUHYnaWxRCwBf6v">
    <img src="https://i.ytimg.com/an_webp/fBUlsuLVDTE/mqdefault_6s.webp?du=3000&sqp=CPKZ5asG&rs=AOn4CLAQ-l7DkMBIoFy6Bmuyb-yrfhNSKw" width="40%">
  </a>
  <br>
  <a href="https://youtu.be/fBUlsuLVDTE?si=vuUHYnaWxRCwBf6v">1차 데모영상</a>
</p>

## 아쉬운 점
-
-

## Reference
-
-

## License
- 
-
