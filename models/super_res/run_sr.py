import argparse
import cv2
import numpy as np
import os
import sys
import torch

from .models.network_swinir import SwinIR as net

def read_img(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_COLOR).astype(np.float32)
    return img

class SRModel:
    def __init__(self, model_path, device="gpu"):
        self.model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'
        pretrained_model = torch.load(model_path)

        self.device = "cuda" if device=="gpu" else "cpu"
        self.model.load_state_dict(
            pretrained_model[param_key_g] 
            if param_key_g in pretrained_model.keys() else pretrained_model,
            strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)

    def predict(self, img_lq, tile=None, tile_overlap=32, window_size=8,
                scale=4):
            if tile is None:
                # predict the image as a whole
                output = self.model(img_lq)
            else:
                # predict the image tile by tile
                b, c, h, w = img_lq.size()
                tile = min(tile, h, w)
                assert tile % window_size == 0, "tile size should be a multiple of window_size"
                sf = scale

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
                W = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch = self.model(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                        W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
                output = E.div_(W)

            return output

    def supersample(self, np_float32_img, tile=None, tile_overlap=32,
                    window_size=8,scale=4):
        if not isinstance(np_float32_img, np.ndarray):
            np_float32_img = np.array(np_float32_img)
        img_lq = np_float32_img/255.0    
        img_lq = np.transpose(
            img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

            output = self.predict(img_lq, tile, tile_overlap, window_size,scale)
            output = output[..., :h_old * scale, :w_old * scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        return output


if __name__=="__main__":
    import sys
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory containing imgs")
    parser.add_argument("--output_dir", default="output", help="Output dir")

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    img_paths = [os.path.join(input_dir,f) for f in os.listdir(input_dir)]
    os.makedirs(output_dir, exist_ok=True)

    tic = time.time()
    sr_model = SRModel("weights/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")
    toc = time.time()    
    print(f'Time to initialize model: {toc-tic:.4f}s')
    
    for img_path in img_paths:
        img = read_img(img_path)
        tic = time.time()
        sr_img = sr_model.supersample(img, tile=400, tile_overlap=32)
        toc = time.time()
        print(f'Time taken to process {os.path.basename(img_path)} : {toc-tic:.3f}s')
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), sr_img)

