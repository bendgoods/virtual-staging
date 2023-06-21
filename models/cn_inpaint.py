import random

import numpy as np
import cv2
import settings.virtual_staging as vs
import torch
from diffusers import ControlNetModel, DDIMScheduler
from PIL import Image
from pipelines.controlnet_inpainting_pipeline import \
    StableDiffusionControlNetInpaintPipeline
from settings.prompt_enhancer import PromptEnhancer
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from utils.ade import ade_palette
from utils.utils import overlay, rounded_rectangle, rescale_image, unpad_image
from .segformer_model import SegFormer
from utils.ade import ADE_CLASSES
import utils.utils as utils

class ControlNetInpaint:
    def __init__(
            self,
            cn_model='lllyasviel/sd-controlnet-seg',
            # cn_model='BertChristiaens/controlnet-seg-room',
            sd_model='runwayml/stable-diffusion-inpainting',
            use_cuda=True
        ) -> None:

        self.use_cuda = use_cuda
        # Device to use
        if use_cuda and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Load all models
        self.load_models(sd_model, cn_model)

    def load_models(self, sd_model, cn_model):
        # Load controlnet
        self.cn_model = ControlNetModel.from_pretrained(
                cn_model,
                torch_dtype=torch.float16
        )
        # Load stable diffusion
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                sd_model,
                controlnet=self.cn_model,
                torch_dtype=torch.float16
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # Memory optimisation
        if self.use_cuda:
            self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.to(self.device)

        # Load the prompt enhancer
        self.prompt_enhancer = PromptEnhancer()

        # Load segmentation models
        self.image_processor = AutoImageProcessor.from_pretrained(
            'openmmlab/upernet-convnext-small'
        )
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            'openmmlab/upernet-convnext-small'
        )

    def get_cn_seg_control(self, image, return_mask=False,
                           dilation_kernel=(5, 5),
                           iterations=10,
                           mask_option='erode'):
        # Pre-process images
        pixel_values = self.image_processor(
                np.array(image),
                return_tensors='pt'
        ).pixel_values

        # Run image segmentation
        with torch.no_grad():
            seg = self.image_segmentor(pixel_values)

        # Refine segmentation
        seg = self.image_processor.post_process_semantic_segmentation(
            seg, target_sizes=[image.size[::-1]]
        )[0]
        class_labels = []
        class_idxs = []

        # If True it will generate a mask for inpainting otherwise segmantation image for controll net will be generated
        if return_mask:            
            mask = seg.cpu().numpy().copy()
            
            # Extracting detected labels and ids
            for i, class_label in enumerate(['floor', 'grass', 'earth', 'path', 'wall']):
                class_idx = ADE_CLASSES.index(class_label)
                if np.any(mask==class_idx):
                    class_labels.append(class_label)
                    class_idxs.append(class_idx)

            # creating a mask of detected objects
            for idx in class_idxs:
                mask[mask==idx] = 255
            mask[mask!=255] = 0

            # Expanding or shinking mask based on `mask_option`
            mask = mask.astype(np.float32)
            kernel = np.ones(dilation_kernel, np.uint8)
            if mask_option == 'erode':
                mask = cv2.erode(mask, kernel, iterations=iterations)
            else:
                mask = cv2.dilate(mask, kernel, iterations=iterations)

            image = Image.fromarray(mask.astype(np.uint8))

        else:  
            # Color code using ADE palette
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(np.array(ade_palette())):
                color_seg[seg==label] = color
            color_seg = color_seg.astype(np.uint8)
            image = Image.fromarray(color_seg)
    
        return image

    def get_prompts(
            self,
            image,
            room_type=None,
            architecture_style=None,
            override_prompt=None
        ):
        # Generate prompts
        prompt, add_prompt, negative_prompt = vs.create_prompts(
                room_type=room_type,
                architecture_style=architecture_style
        )
        prompt = prompt + ', ' + add_prompt

        # Override prompt if user has specified
        if override_prompt:
            prompt = override_prompt

        # Enhance prompt
        prompt = random.choice(self.prompt_enhancer(prompt))

        return prompt, negative_prompt

    def run_model(
            self,
            prompt,
            negative_prompt,
            image,
            mask,
            controlnet_image,
            num_inference_step,
            guidance_scale,
            control_strength,
            generator
        ):
        # Text prompt
        input_ids = self.pipe.tokenizer(
                prompt,
                return_tensors='pt'
        ).input_ids.to(self.device)
        negative_ids = self.pipe.tokenizer(
                negative_prompt,
                truncation=False,
                padding='max_length',
                max_length=input_ids.shape[-1],
                return_tensors='pt'
        ).input_ids.to(self.device)

        # Encode prompts in chunks because of max_length limit
        concat_embeds, neg_embeds = [], []
        max_length = self.pipe.tokenizer.model_max_length
        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(
                    self.pipe.text_encoder(input_ids[:,i:i+max_length])[0]
            )
            neg_embeds.append(
                    self.pipe.text_encoder(negative_ids[:,i:i+max_length])[0]
            )

        # Concat chunks
        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

        # Run through pipe
        images = self.pipe(
                image=image,
                mask_image=mask,
                control_image=controlnet_image,
                num_images_per_prompt=1,
                num_inference_steps=num_inference_step,
                guidance_scale=guidance_scale,
                generator=generator,
                controlnet_conditioning_scale=control_strength,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds
        ).images
        return images

    def generate_mask(self, image, mask_dilation,
                       mask_option, use_rounded=False):
        """Function to only run and display SegFormer for mask debugging.
            integrated with `Generate Mask` button in gradio app.
        """
        if isinstance(image, dict):
            image = image['image'].convert('RGB')
        
        mask = self.get_mask(image, mask_dilation, mask_option=mask_option,
                             use_rounded=use_rounded)
        mask = Image.fromarray(overlay(np.array(image),
                                       np.array(mask),
                                       (255, 0, 0), 0.5))
        return [mask]

    def get_mask(self, image, mask_dilation,
                  mask_option, use_rounded=False):
        W, H = image.size
        # if isinstance(image, np.ndarray):
        #     image = np.array(image.convert('RGB'))
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        smask = self.get_cn_seg_control(image, return_mask=True,
                                iterations=mask_dilation,
                                mask_option=mask_option)

        if use_rounded:
            padding = 10
            smask = np.array(smask)
            mask = rounded_rectangle(np.zeros_like(smask), (0 + padding, 0 + padding),
                                     (H - padding, W - padding), radius=0.5, color=(255, 255, 255), thickness=-1)
            mask = mask//255
            smask = smask//255
            smask = mask*smask
            smask *= 255    
            smask = Image.fromarray(smask)
        return smask

    def __call__(
            self,
            image_dict,
            room_type,
            architecture_style=None,
            negative_prompt="",
            num_images_per_prompt=5,
            guidance_scale=12,
            num_inference_step=20,
            strength_min=0.1,
            strength_max=0.5,
            seed=0,
            override_prompt=None,
            upscale=False,
            mask_dilation=1,
            mask_option='dilate',
            use_fixed_strength=False,
            use_rounded=True,
            pad=True
        ):
        W, H = image_dict['image'].size
        org_image = image_dict['image'].copy()
        input_image = image_dict['image'].convert('RGB')
        input_image = Image.fromarray(utils.resize_image(np.array(input_image), 512))
        rW, rH = input_image.size
        print(input_image.size)
        # pad = True
        # input_image, (rW, rH) = rescale_image(input_image, pad=pad)
        # input_image = image_dict['image'].convert('RGB').resize((512, 512))
        mask_image = image_dict['mask'].convert('RGB')
        mask_image = cv2.resize(
                        np.array(mask_image), (rW, rH), interpolation=cv2.INTER_NEAREST)
        mask_image = Image.fromarray(mask_image)
        
        # mask_image, _ = rescale_image(mask_image, pad=pad)
        # mask_image = image_dict['mask'].convert('RGB').resize((512, 512))


        if not np.any(np.array(mask_image)):
            temp_inp = unpad_image(input_image, (rW, rH))
            print('Extracting Mask...')
            mask_image = self.get_mask(temp_inp, mask_dilation,
                                        mask_option=mask_option,
                                        use_rounded=use_rounded)
            # mask_image, _ = rescale_image(mask_image, pad=pad)
            # mask_image = mask_image.resize((512, 512))

        mask_image = cv2.resize(
                        np.array(mask_image), (rW, rH), interpolation=cv2.INTER_NEAREST)
        mask_image = Image.fromarray(mask_image)
        print(mask_image.size)

        strength_factor = (strength_max - strength_min)/num_images_per_prompt
        control_strength = strength_max
        output_images = []
        for i in range(num_images_per_prompt):
            # increase control strength iteratively
            if not use_fixed_strength:
                control_strength = strength_min + (i+1)*strength_factor

            # Get prompts
            prompt, negative_prompt = self.get_prompts(
                    input_image,
                    room_type,
                    architecture_style=architecture_style,
                    override_prompt=override_prompt
            )

            # Get control image
            temp_inp = unpad_image(input_image, (rW, rH))
            control = self.get_cn_seg_control(image=temp_inp.copy())
            control = cv2.resize(
                        np.array(control), (rW, rH), interpolation=cv2.INTER_NEAREST)
            control = Image.fromarray(control)
            print(control.size)
  
            # control, _ = rescale_image(control, pad=pad)
            # Set seed
            if seed == 0:
                itr_seed = torch.randint(0, 1000000, (1,))
            else:
                itr_seed = seed
            itr_seed_gen = torch.manual_seed(itr_seed)

            print(f'\tUsing model with seed {itr_seed} '
                  f'strength {control_strength} '
                  f'and guidance {guidance_scale} '
                  f'with prompt: \n\t{prompt}')

            # Run model
            output = self.run_model(
                prompt,
                negative_prompt,
                input_image,
                mask_image,
                control,
                num_inference_step,
                guidance_scale,
                control_strength,
                itr_seed_gen
            )
            output_images.append(output[0].copy())
            input_image = output[0].copy()

        # Resize input image to match output image
        W, H = output_images[0].size[:2]
        
        input_image = np.array(unpad_image(
                                rescale_image(org_image.convert('RGB'),
                                             size=(W,H), pad=pad)[0],
                                (rW, rH)))
        # Resize mask to match input image
        mask_image = np.array(unpad_image(
                                rescale_image(mask_image.convert('L'),
                                            size=(W, H), pad=pad)[0],
                                (rW, rH)))

        # Overlay mask on input image
        mask = [Image.fromarray(overlay(input_image, mask_image,
                                        (255, 0, 0), 0.5))]
        output_images = [unpad_image(img, size=(rW, rH)) for img in output_images]
        # mask_image_postproc = utils.convolution(Image.fromarray(mask_image).convert('RGB'), size=41)
        
        # for i in range(len(output_images)):
        #     output_images[i] = Image.composite(output_images[i].convert("RGBA"),
        #                                        Image.fromarray(input_image).convert("RGBA"),
        #                                        Image.fromarray(mask_image)
        #                                     ).convert('RGB')

        
        if upscale:
            output_images = [rescale_image(img, size=org_image.size,
                                           pad=False)[0] for img in output_images]
            mask = [rescale_image(m, size=org_image.size,
                                           pad=False)[0] for m in mask]

        return output_images, mask
