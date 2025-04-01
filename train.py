import warnings
from ultralytics import RTDETR, YOLO

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    model = RTDETR('/\\ultralytics\\cfg\\models\\rt-detr\\rtdetr-EVT-EAA-PSCONV-WTCONV-CAFM.yaml')

    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=40,
                batch=1,
                workers=2,
                device='0',
                project='runs/train',
                name='EEC-DETR',
                )
