from share import *

from cldm.model import create_model, load_state_dict
import cv2
from annotator.util import resize_image
import numpy as np
import torch
import einops
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
from annotator.util import HWC3
import random
from pytorch_lightning import seed_everything
from annotator.canny import CannyDetector

apply_canny = CannyDetector()


def load_image_as_array(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        image_array = np.array(img)
        return image_array


# Configs
# resume_path = '/notebooks/erase/stable-diffusion/controlnet_files/control_sd15_canny.pth' # your checkpoint path
resume_path = "/notebooks/erase/stable-diffusion/models/compvis-word_bird-method_xattn-sg_3-ng_1-iter_500-lr_1e-05/compvis-word_bird-method_xattn-sg_3-ng_1-iter_500-lr_1e-05.pt"
N = 1
ddim_steps = 50


model = create_model('/notebooks/erase/stable-diffusion/controlnet_files/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model = model.cuda()
sampler = DDIMSampler(model)
output_img_path = 'canny_bird_output.png'

a_prompt = "best quality, extremely detailed"
n_prompt = "longbody, lowres,bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
prompt = "Bird"
guess_mode = False
n_samples=1
h = 512
w = 512

low_threshold = 100
high_threshold = 200

# For the conditional image init
cond_img = load_image_as_array("/notebooks/erase/stable-diffusion/test_images/images_bird_canny.png")
cond_img = resize_image(HWC3(cond_img), 512)

h, w, c = cond_img.shape

# cond_detected_map = np.zeros_like(cond_img, dtype=np.uint8)
# cond_detected_map[np.min(cond_img, axis=2) < 127] = 255

cond_detected_map = apply_canny(cond_img, low_threshold, high_threshold)
cond_detected_map = HWC3(cond_detected_map)

cond_control = torch.from_numpy(cond_detected_map.copy()).float().cuda() / 255.0
cond_control = torch.stack([cond_control for _ in range(n_samples)], dim=0)
cond_control = einops.rearrange(cond_control, 'b h w c -> b c h w').clone()

seed = random.randint(0, 65535)
seed_everything(0)

cond = {"c_concat": [cond_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * n_samples)]}
un_cond = {"c_concat": None if guess_mode else [cond_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * n_samples)]}

shape = (4, h//8, w//8)

model.control_scales = [1 * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([1.0] * 13)

samples, inters = sampler.sample(ddim_steps, n_samples, shape, cond, verbose=False, eta=0.0, unconditional_guidance_scale=9.0, unconditional_conditioning=un_cond)


print("samples: ", samples.shape)
x_samples = model.decode_first_stage(samples)
x_samples = x_samples.squeeze(0)
x_samples = (x_samples + 1.0) / 2.0
x_samples = x_samples.transpose(0, 1).transpose(1, 2)
x_samples = x_samples.cpu().numpy()
x_samples = (x_samples * 255).astype(np.uint8)

image_name = output_img_path.split('/')[-1]
Image.fromarray(x_samples).save('./outputs/' + image_name)
