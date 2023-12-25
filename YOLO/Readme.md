# Custom Training, Evaluation, and Inference with YOLOv5 & YOLOv8

This repository provides comprehensive guidelines for training, evaluating, and performing inference with YOLOv5 and YOLOv8 models on custom datasets.

## Prerequisites
- Python 버전
- CUDA (for GPU support)

## Installation
-yolov5 가상환경 생성
git clone https://github.com/ultralytics/yolov5.git
pip install ultralytics 
cd yolov5
pip install -r requirements.txt

-yolov8 가상환경 생성
git clone https://github.com/ultralytics/ultralytics.git
pip install ultralytics 

## Custom Dataset Preparation

2. 모델 학습
커스텀 데이터셋 train,test,val로 나누고 yaml파일 작성
pre-trained 모델으로 모델 학습

python segment/train.py --data './dataset/line_label/YOLODataset/dataset.yaml' --weights yolov5n-seg.pt --img 640 (yolov5 - cli환경)


from ultralytics import YOLO
model = YOLO("yolov8n-seg.pt")  # load a pretrained model 
model.train(data="./data/line_label/YOLODataset/dataset.yaml", epochs=100) (yolov8 - python환경)



3. 모델 추론
python segment/predict.py --weights /home/hangyu/dev_ws/yolov5/yolov5/runs/train-seg/exp/weights/best.pt --source 0 (yolov5)

yolo predict model=./runs/segment/train6/weights/best.pt source=0 show=True (yolov8)
