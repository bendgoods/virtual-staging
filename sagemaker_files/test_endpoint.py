import boto3
import cv2
import sys
sys.path.append("./")

from basic_utils.utils import load_config

runtime = boto3.client(service_name='sagemaker-runtime')

config = load_config('sagemaker_files/config.yaml')
ENDPOINT_NAME = config.ENDPOINT_NAME

# test using image file
orig_image = cv2.imread('test_imgs/v1/Balcony_73.jpg')
    
payload = cv2.imencode('.png', orig_image)[1].tobytes()
response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, 
                                   ContentType='multipart/form-data', 
                                   Body=payload)

print(response['Body'].read().decode())

# test using image url
payload = "https://images.livspace-cdn.com/plain/https://d3gq2merok8n5r.cloudfront.net/abhinav/ond-1634120396-Obfdc/jfm-2023-1672723560-eFGVH/balcony-1674210268-39YjC/bll-1675062554-QvyMI.jpg"
runtime = boto3.client(service_name='sagemaker-runtime')
response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, 
                                   ContentType='multipart/form-data', 
                                   Body=payload)

print(response['Body'].read().decode())