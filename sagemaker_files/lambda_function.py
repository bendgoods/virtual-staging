import os
import io
import boto3
import json
import base64
import logging

# grab runtime client
runtime = boto3.client("sagemaker-runtime")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):

    ENDPOINT_NAME='watermark-text-detection'

    data = json.loads(json.dumps(event))

    if  "image_url" in data.keys():
        payload = data['image_url']
    else:        
        image = data['image'].encode()
        payload = base64.b64decode(image)

    logger.info(f"payload: {type(payload)}")

    # Invoke the model. In this case the data type is a file or str
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                   ContentType='multipart/form-data',
                                   Body=payload)
    
    # Get the body of the response from the model
    result = response['Body'].read().decode()
    # Return it along with the status code of 200 meaning this was succesful 
    return {
        'statusCode': 200,
        'body': result,
    }