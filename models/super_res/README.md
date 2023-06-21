# Installation
1. Install latest version of pytorch
2. Install requirement in requirements.txt file
# Download
Download the weights from google drive [here](https://drive.google.com/file/d/1GtsTqtZh5FBStky1Z5v6aDjYMBiyOcgR/view?usp=sharing)
or
```
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth -P weights
```
# Integrate
```python
from run_sr import SRModel

sr_model = SRModel("weights/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")
img = cv2.imread(img_path,cv2.IMREAD_COLOR).astype(np.float32)
sr_img = sr_model.supersample(img, tile=400, tile_overlap=32)
```
