import numpy as np
import torch
import json
import io
from models.inference import RoomClassifier
import logging
from PIL import Image
import requests
import cv2

logger = logging.getLogger("sagemaker-inference")

def model_fn(model_dir):
    """Function to load model"""

    logger.info("Executing model_fn from inference.py...")
    logger.info(f"model_dir: {model_dir}")

    model = model = RoomClassifier(device=True)

    return model

def input_fn(request_body, request_content_type):
    """Function to get input from api"""

    logger.info("Executing input_fn from inference.py ...")

    if not request_content_type:
        raise Exception("Unsupported content type: " + request_content_type)

    try:
        logger.info(type(request_body.decode()))
        logger.info('Retrieving image from:', request_body.decode())
        response = requests.get(request_body.decode())
        bytes_im = io.BytesIO(response.content)
        logger.info(type(bytes_im))
        img = np.array(Image.open(bytes_im))
    except:       
        logger.info(type(request_body))
        jpg_as_np = np.frombuffer(request_body, 
		                          dtype=np.uint8)
        logger.info(type(jpg_as_np))
        img = cv2.imdecode(jpg_as_np, flags=-1)

    logger.info(type(img))
    return img

def predict_fn(input_data, model):
    """Function to run model inference"""

    logger.info("Executing predict_fn from inference.py ...")
    result = model(input_data, k=5)

    return result

def output_fn(prediction_output, content_type):
    """Function to return output to api"""

    print(prediction_output)
    result = {
    "RoomTypes" : prediction_output
    }
    
    logger.info("Executing output_fn from inference.py ...")
    return json.dumps(result)