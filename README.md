# Virtual Staging

1. Clone repo

```
git clone https://github.com/axcelerateai/virtual_staging.git
cd virtual_staging
```

2. Install requirements
```
python3 -m venv .env
source .env/bin/activate

pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

3. Download upsampling weights

```
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth -P weights
```

4. Run Gradio App

```
python cn__gradio.py
```

5. Use FastAPI

```
python app.py
```