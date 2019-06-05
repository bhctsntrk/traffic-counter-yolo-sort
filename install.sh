#/bin/bash

export fileid=10F4n3uFXMmGfCzZ-U_Etqk7Lk22B3mm2
export filename=./yolo-coco/yolov3-aerial.weights

wget -O /tmp/gdrivedl 'https://f.mjh.nz/gdrivedl'
chmod +x /tmp/gdrivedl
/tmp/gdrivedl ${fileid} ${filename}

sudo apt install python3-tk python3-pip
pip3 install -r requirements.txt --user
