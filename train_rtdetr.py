import sys
import argparse
import os

sys.path.append(r'E:\GitHubRepo\PR\ultralyticsPro-') # Path

from ultralytics import RTDETR

'''
RT-DETR训练示例
python train_rtdetr.py --cfg ultralytics\cfg\models\rt-detr\rtdetr-l.yaml 

'''

def main(opt):
    yaml = opt.cfg
    weights = opt.weights
    model = RTDETR(weights)

    model.info()

    results = model.train(data='coco128.yaml', 
                        epochs=100, 
                        imgsz=640, 
                        workers=0, 
                        batch=2,
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'ultralytics\cfg\models\rt-detr\rtdetr-l.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='rtdetr-l.pt', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)