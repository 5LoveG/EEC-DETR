# EEC-DETR
## SAR ship detection

### Requirements
1. First, install the `ultralytics` package in a Python environment with `Python>=3.8` and `PyTorch>=1.8`.
```bash
pip install ultralytics
```
2. Then, install the required environment for this project.
```bash
pip install requirements.txt
```
3. Select various model configurations in the `cfg\models` folder for experimental comparison. There are already baseline models and the EEC-DETR model designed by me. The `.py` files in the `nn\modules` folder contain various modules, such as `PSCONV`, `WTCONV`, etc. You are supported to create new `.yaml` files in the `cfg\models` folder to freely configure new models. 


