import random

import numpy as np
import settings.virtual_staging as vs
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from settings.prompt_enhancer import PromptEnhancer
from utils.utils import overlay

from .segformer_model import SegFormer


class StableDiffusionInpaint:
    def __init__(
            self,
            model_path='runwayml/stable-diffusion-inpainting',
            use_cuda=True
        ):
        # Set device
        if self.use_cuda and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Load model
        self.pipe = self.load_model(model_path)


    def load_model(self, model_path):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path)
        self.pipe.to(self.device)
        if self.use_cuda:
            self.pipe.enable_xformers_memory_efficient_attention()
        self.prompt_enhancer = PromptEnhancer()

        return self.pipe

    def get_staging_prompts(
            self,
            input_image,
            room_type=None,
            architecture_style=None
        ):
        # Generate prompts
        prompt, add_prompt, negative_prompt = vs.create_prompts(
                room_type=room_type,
                architecture_style=architecture_style
        )
        return prompt, add_prompt, negative_prompt

    def _get_outputs(self, prompt: str, image, mask,
                      num_inference_step: int,
                      guidance_scale: float, generator: int):
        # 2. Forward embeddings and negative embeddings through text encoder

        max_length = self.pipe.tokenizer.model_max_length

        input_ids = self.pipe.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        negative_ids = self.pipe.tokenizer("",
                                            truncation=False,
                                            padding="max_length",
                                            max_length=input_ids.shape[-1],
                                            return_tensors="pt").input_ids
        negative_ids = negative_ids.to(self.device)

        concat_embeds = []
        neg_embeds = []
        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(self.pipe.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(self.pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

        # 3. Forward
        images = self.pipe(image=image,
                        mask_image=mask,
                        num_images_per_prompt=1,
                        num_inference_steps=num_inference_step,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds).images
        return images

    def get_mask(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        seg_model = SegFormer(classes=['floor', 'grass', 'earth', 'path'])
        mask = seg_model(image)
        return mask

    def generate_image(
        self,
        pil_image: str,
        room_type: str,
        architecture_style: str,
        negative_prompt: str,
        num_images_per_prompt: int,
        guidance_scale: int,
        num_inference_step: int,
        seed_generator=0,
        override_prompt=None
    ):
        """Function to generate image"""
        # get image size
        iw, ih = pil_image["image"].size

        # rescale image image and mask
        image = pil_image["image"].convert("RGB").resize((512, 512))
        mask_image = pil_image["mask"].convert("RGB").resize((512, 512))

        if not np.any(np.array(mask_image)):
            print('Extracting Mask...')
            mask_image = self.get_mask(image)

        images = []
        for _ in range(num_images_per_prompt):
            architecture_style = None
            if seed_generator == 0:
                random_seed = torch.randint(0, 1000000, (1,))
                generator = torch.manual_seed(random_seed)
            else:
                generator = torch.manual_seed(seed_generator)

            # Customise to mode
            prompt, add_prompt, negative_prompt = self.get_staging_prompts(
                    image,
                    room_type,
                    architecture_style
            )

            prompt = prompt + ', ' + add_prompt

            if override_prompt:
               prompt = override_prompt

            prompt = random.choice(self.prompt_enhancer(prompt))

            # Run model
            print(f'Using model with '
                    f'guidance {guidance_scale} with prompt\n: {prompt}')

            output = self._get_outputs(prompt=prompt, image=image,
                                        mask=mask_image,
                                        num_inference_step=num_inference_step,
                                        guidance_scale=guidance_scale,
                                        generator=generator
                                        )
            images.append(output[0])

        output = [img.resize((iw, ih)) for img in images]
        image = np.array(image.resize((iw, ih)))
        mask_image = np.array(mask_image.convert('L').resize((iw, ih)))

        # resize geenrated images to original size
        mask = [Image.fromarray(overlay(image,
                                        mask_image, (255, 0, 0), 0.5))]
        return output, mask
