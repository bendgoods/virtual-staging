import cv2
import torch
import numpy as np
from PIL import Image

class Upsampler:
    def __init__(
            self,
            model_type='edsr',
            model_path='weights/EDSR_x4.pb',
            scale=4,
            prefer_cuda=True
        ) -> None:

        self.model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model.readModel(model_path)
        self.model.setModel(model_type, scale)

        if prefer_cuda and torch.cuda.is_available():
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def upsample(self, image):
        if Image.isImageType(image):
            image = np.array(image)

        result = self.model.upsample(image)
        return Image.fromarray(result)

    def __call__(self, input_image):
        # Check if single input image or multiple
        if isinstance(input_image, list):
            return_as_list = True
        else:
            input_image = [input_image]
            return_as_list = False

        output_images = []
        for image in input_image:
            image = self.upsample(image)
            output_images.append(image)

        return output_images if return_as_list else output_images[0]
