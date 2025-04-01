from ultralytics import RTDETR, YOLO

model_path = r""
img_dir = r""

model = RTDETR(model_path)

model.predict(img_dir, save=True, save_txt=False, show_conf=False, show_labels=False)
