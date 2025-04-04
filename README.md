# EEC-DETR
## SAR ship detection

Xinchi Zhao "Efficient Transformer - based SAR Ship Detection with Hybrid Cross - Fusion Modules," The Visual Computer

### Requirements
1. First, install the `ultralytics` package in a Python environment with `Python>=3.8` and `PyTorch>=1.8`.
```bash
pip install ultralytics
```
2. Then, install the required environment for this project.
```bash
pip install requirements.txt
```
### Usage
1. Select various model configurations in the `cfg\models` folder for experimental comparison. There are already baseline models and the EEC-DETR model designed by me. The `.py` files in the `nn\modules` folder contain various modules, such as `PSCONV`, `WTCONV`, etc. You are supported to create new `.yaml` files in the `cfg\models` folder to freely configure new models. 

2. train example
```
# Import the RTDETR class from the ultralytics library
from ultralytics import RTDETR

if __name__ == '__main__':
    # Initialize an RTDETR model with a specified configuration file
    # The path points to the configuration file for the model
    model = RTDETR('/\\ultralytics\\cfg\\models\\rt-detr\\rtdetr-EVT-EAA-PSCONV-WTCONV-CAFM.yaml')

    # Train the model with the specified parameters
    model.train(
        # Path to the data configuration file
        data='dataset/data.yaml',
        # Whether to cache the dataset. Here it's set to False
        cache=False,
        # Image size for training
        imgsz=640,
        # Number of training epochs
        epochs=40,
        # Batch size for training
        batch=1,
        # Number of worker threads for data loading
        workers=2,
        # Device to use for training. Here it's set to GPU 0
        device='0',
        # Project directory to save the training results
        project='runs/train',
        # Name of the experiment
        name='EEC-DETR'
    )

```
3. predict example
```
from ultralytics import RTDETR, YOLO

# Define the path to the model
model_path = r""
# Define the directory containing the images
img_dir = r""

# Initialize the RTDETR model with the specified model path
model = RTDETR(model_path)

# Use the model to perform prediction on the images in the given directory
# save=True: Save the prediction results
# save_txt=False: Do not save the prediction results in text format
# show_conf=False: Do not show the confidence scores
# show_labels=False: Do not show the labels
model.predict(img_dir, save=True, save_txt=False, show_conf=False, show_labels=False)
```



