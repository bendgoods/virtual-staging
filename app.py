import io
import logging
import os
import sys
import traceback
from typing import Optional

sys.path.append('controlnet')
# isort: split

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import StreamingResponse

from models.cn_inpaint import ControlNetInpaint

app = FastAPI(title='Virtual Staging')

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg']
room_types = ['kitchen', 'bedroom', 'bathroom', 'living room', 'backyard']
architecture_style = [
        'coastal',
        'rustic',
        'modern',
        'industrial',
        'scandinavian',
        'french provincial'
]

def write_text_on_blank_image(text, img_shape, scale=1):
    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0]
    img_shape[0] = textsize[1] + 20
    X_text = int((img_shape[1] - textsize[0]) / 2)
    Y_text = int((img_shape[0] + textsize[1]) / 2)
    img = np.zeros(img_shape, np.uint8)
    cv2.putText(img, text, (X_text, Y_text), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 3, 0)
    return img

def concat_images(images_list):
    # Divide into columns of equal length
    col_1 = images_list[::2]
    col_2 = images_list[1::2]
    if len(col_2) < len(col_1):
        col_2.append(255*np.ones_like(col_1[0])) # at most one difference
    assert len(col_1) == len(col_2)

    # Concat
    concat = np.vstack([np.hstack([c1,c2]) for c1,c2 in zip(col_1,col_2)])
    return concat

def process_image(
        model,
        input_image,
        mask,
        room_type,
        architecture_style=None,
        negative_prompt="",
        num_iterations=4,
        guidance_scale=15,
        num_inference_step=20,
        strength_min=0.1,
        strength_max=0.5,
        seed=0,
        override_prompt=None,
        upscale=False,
        mask_dilation=10,
        use_fixed_strength=True,
        use_rounded=False,
        debug=False
    ):
   
    logging.info(f"Processing: Virtual Room Staging")
    image_dict = {
        'image': input_image,
        'mask': mask
    }
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
    if not isinstance(output_images[0], np.ndarray):
        output_images = [cv2.cvtColor(np.array(image),
                                       cv2.COLOR_RGB2BGR) for image in output_images]
    if debug:
        # Concat images
        h, w = output_images[0].shape[:2]
        if not isinstance(input_image, np.ndarray):
            input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        if not isinstance(mask[0], np.ndarray):
            mask = cv2.cvtColor(np.array(mask[0]), cv2.COLOR_RGB2BGR)
        input_image_resized = cv2.resize(input_image, (w,h))
        # mask = np.array(mask)
        output = concat_images([input_image_resized, mask] + list(output_images))
        # output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    else:
        output = output_images[-1]
    return output

@app.on_event('startup')
def init():
    global model
    # Loading models
    model = ControlNetInpaint()

@app.get("/health")
def check_health():
    logging.info("Checking health")
    return ({"Message": 'System is up!'})

@app.post('/generate-mask')
def generate_mask(
            image: UploadFile=File(...),
            debug: bool=True,
            mask_dilation: int = 10,
            use_rounded: bool=True
            ):
    image = Image.open(image.file)
    if debug:
        mask = model.generate_mask(image, mask_dilation, mask_option='erode', use_rounded=use_rounded)[0]
    else:
        mask = model.get_mask(image, mask_dilation, mask_option='erode', use_rounded=use_rounded)
    if not isinstance(mask, np.ndarray):
        mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)

    res, im_png = cv2.imencode(".png", mask)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

@app.post('/virtual-staging')
def generate_image(
        room_type: str = Form(..., description='Select a room type', enum=room_types),
        architecture_style: str = Form(..., description='Select an architecture style', enum=['None']+architecture_style),
        num_iterations: int = Form(4, description='Number of interation to perform'),
        image: UploadFile=File(...),
        mask: UploadFile=None,
        override_prompt: Optional[str]=None,
        debug: bool=True,
        upscale: bool=True,
        use_rounded: bool=True
    ):
    logging.info(f'{image.filename}')
    extention = os.path.splitext(image.filename)[1]

    if extention in ALLOWED_EXTENSIONS:
        img = Image.open(image.file)
        if mask:
            mask = Image.open(mask.file)
        else:
            mask = Image.fromarray(np.zeros_like(np.array(img)))

        if architecture_style == 'None':
            architecture_style = None
        img = process_image(
                model,
                img,
                mask,
                room_type,
                architecture_style=architecture_style,
                num_iterations=num_iterations,
                override_prompt=override_prompt,
                debug=debug,
                upscale=upscale,
                use_rounded=use_rounded
        )
        res, im_png = cv2.imencode(".png", img)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    else:
        logging.error(f"Traceback: {traceback.format_exc()}")
        logging.info("Request finished")
        raise HTTPException(status_code=500, detail="Allowed extentions are : 'png', 'jpg', 'jpeg'")


if __name__=='__main__':
    port = 7772
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    uvicorn.run('app:app', host='0.0.0.0', port=port, reload=False)
