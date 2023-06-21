import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from utils.ade import ADE_CLASSES


class SegFormer:
    def __init__(
            self,
            classes,
            device='cpu',
            model_name='nvidia/segformer-b5-finetuned-ade-640-640'
        ) -> None:
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.model.to(self.device)
        self.CLASSES = classes

    def __call__(self, image, mask_option='dilate', dilation_kernel=(9, 9), iterations=0):
        inputs = self.extractor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = nn.functional.interpolate(outputs.logits.detach().cpu(),
                        size=image.shape[:-1], # (height, width)
                        mode='bilinear',
                        align_corners=False)
        seg = logits.argmax(dim=1)[0]
        seg = seg.numpy()
        result = seg.copy()

        class_labels = []
        class_idxs = []
        for i, class_label in enumerate(self.CLASSES):
            class_idx = ADE_CLASSES.index(class_label)
            if np.any(result==class_idx):
                class_labels.append(class_label)
                class_idxs.append(class_idx)

        for idx in class_idxs:
            result[result==idx] = 255
        result[result!=255] = 0

        result = result.astype(np.float32)
        kernel = np.ones(dilation_kernel, np.uint8)
        if mask_option == 'erode':
            result = cv2.erode(result, kernel, iterations=iterations)
        else:
            result = cv2.dilate(result, kernel, iterations=iterations)

        return Image.fromarray(result.astype(np.uint8))

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])

    arch = ADE_CLASSES

    segformer = SegFormer(arch)
    mask = segformer(img)
    mask.show()
