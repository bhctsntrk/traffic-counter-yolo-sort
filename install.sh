#/bin/bash

sudo apt install python3-tk python3-pip
wget -O ./yolo-coco/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
pip3 install -r requirements.txt --user
