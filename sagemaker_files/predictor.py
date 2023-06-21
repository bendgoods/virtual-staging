import io
import logging
import os
import sys
from typing import Optional

sys.path.append('.')
# isort: split

import uvicorn
from fastapi import FastAPI, Request, Response
from PIL import Image
import base64
import requests

from models.cn_inpaint import ControlNetInpaint
import json
import numpy as np
import cv2

app = FastAPI()

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CWD = os.path.abspath(os.getcwd())

@app.on_event('startup')
def init():
    global model
    # Loading models
    model = ControlNetInpaint()

def generate_mask(image, mask_dilation=10, use_rounded=True):
    mask = model.get_mask(image, mask_dilation, mask_option='erode', use_rounded=use_rounded)
    if not isinstance(mask, np.ndarray):
        mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
    return mask


@app.get("/ping")
def ping_check():
    logger.info("PING!")
    return Response(content=json.dumps({"ping_status": "ok"}), status_code=200)

@app.post('/invocations')
@app.put('/invocations')
async def handler(req: Request):
    try:
        model_inputs = await req.json()
        if not isinstance(model_inputs, dict):
            model_inputs = json.loads(model_inputs)
        room_type = model_inputs["room_type"]    
        architecture_style = model_inputs.get("architecture_style", None)            
        upscale = model_inputs.get("upscale", False)            
        return_mask = model_inputs.get("return_mask", False)            
        
        negative_prompt = model_inputs.get("negative_prompt", "")
        num_iterations = int(model_inputs.get("num_iterations", 4))
        guidance_scale = float(model_inputs.get("guidance_scale", 12.4))
        num_inference_step = int(model_inputs.get("num_inference_step", 20))
        strength_min = float(model_inputs.get("strength_min", 0.1))
        strength_max = float(model_inputs.get("strength_max", 0.5))
        seed = model_inputs.get("seed", 0)
        override_prompt = model_inputs.get("override_prompt", None)
        mask_dilation = int(model_inputs.get("mask_dilation", 10))
        use_fixed_strength = model_inputs.get("use_fixed_strength", True)
        use_rounded = model_inputs.get("use_rounded", True)
        debug = model_inputs.get("debug", False)

        # If image URL is provided, download the image
        if 'image_url' in model_inputs.keys():
                logger.info("Image Url section")
                image_url = model_inputs['image_url']
                logger.info(f"image url: {image_url}")
                logger.info(f"image url type: {type(image_url)}")
                response = requests.get(image_url)
                input_img = Image.open(io.BytesIO(response.content))
                
        elif 'image' in model_inputs.keys():
            # If image is provided as base64 string, decode it
            logger.info("image")
            img_data = model_inputs['image']

            logger.info(f"image: {img_data}")
            logger.info(f"image: {type(img_data)}")
            # img_data = img_data.encode()
            img_binary = base64.b64decode(img_data)
            input_img = Image.open(io.BytesIO(img_binary))
        else:
            return json.dumps({'error': 'Image URL or base64-encoded image required'})

        # get mask of image
        if 'mask_url' in model_inputs.keys():
            logger.info("Mask Url section")
            mask_url = model_inputs['mask_url']
            logger.info(f"mask url: {mask_url}")
            logger.info(f"mask url type: {type(mask_url)}")
            response = requests.get(mask_url)
            mask = Image.open(io.BytesIO(response.content))
        elif 'mask' in model_inputs.keys():
            # If image is provided as base64 string, decode it
            logger.info("mask")
            mask_data = model_inputs['mask']

            logger.info(f"mask: {mask_data}")
            logger.info(f"mask: {type(mask_data)}")
            mask_binary = base64.b64decode(mask_data)
            mask = Image.open(io.BytesIO(mask_binary))
        else:
            mask = Image.fromarray(
                np.zeros_like(np.array(input_img.copy())))
        
        image_dict = {
                'image': input_img,
                'mask' : mask
                }
        if return_mask:
            output = generate_mask(input_img, mask_dilation, use_rounded)
        else:  
            output_images, mask = model(image_dict,
                                        room_type,
                                        architecture_style=architecture_style,
                                        negative_prompt=negative_prompt,
                                        num_images_per_prompt=num_iterations,
                                        guidance_scale=guidance_scale,
                                        num_inference_step=num_inference_step,
                                        strength_min=strength_min,
                                        strength_max=strength_max,
                                        seed=seed,
                                        override_prompt=override_prompt,
                                        upscale=upscale,
                                        mask_dilation=mask_dilation,
                                        mask_option='erode',
                                        use_fixed_strength=use_fixed_strength,
                                        use_rounded=use_rounded
                                    )      
            output = np.array(output_images[-1])
        
        res, im_png = cv2.imencode(".png", cv2.cvtColor(output,
                                                        cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(im_png).decode('utf-8')
        return Response(status_code=200, content=json.dumps({"result": img_str}))
    except Exception as e:
        return Response(status_code=500, content=json.dumps({"error": f"{e}"}))

if __name__=='__main__':
    port = 7771
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    uvicorn.run('predictor:app', host='0.0.0.0', port=port, reload=True)
