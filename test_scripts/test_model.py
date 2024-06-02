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


object_class = "fish"
input_type = "canny"
model_used = "scribble"
unconditional_guidance_scale = 7
res = 512

# Configs
# resume_path = '/notebooks/erase/stable-diffusion/models/ldm/controlnet_scribble/control_sd15_scribble.pth' # your checkpoint path
resume_path = "/workspace/compvis-word_airplane-method_notime-sg_3-ng_1-iter_500-lr_1e-05_scribble.pt"
N = 1
ddim_steps = 50


model = create_model('/root/Steering-Diffusion/configs/controlnet/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cuda'), strict=False)
model = model.cuda()
sampler = DDIMSampler(model)
output_img_path = f'{object_class}-prompt_{object_class}-{input_type}_{unconditional_guidance_scale}-gs_{model_used}-notime.png'
# output_img_path = 'fish_test.png'

a_prompt = "best quality, extremely detailed"
n_prompt = "worst quality, low quality, noisy"
prompt = object_class
guess_mode = False
n_samples=1
h = res
w = res

low_threshold = 100
high_threshold = 200

# For the conditional image init
cond_img = load_image_as_array(f"../test_input_images/{object_class}_{input_type}.png")
cond_img = resize_image(HWC3(cond_img), res)

h, w, c = cond_img.shape

# cond_detected_map = np.zeros_like(cond_img, dtype=np.uint8)
# cond_detected_map[np.min(cond_img, axis=2) < 127] = 255

# cond_detected_map = apply_canny(cond_img, low_threshold, high_threshold)
# cond_detected_map = HWC3(cond_detected_map)

cond_detected_map = np.zeros_like(cond_img, dtype=np.uint8)
cond_detected_map[np.min(cond_img, axis=2) < 127] = 255

cond_control = torch.from_numpy(cond_detected_map.copy()).float().cuda() / 255.0
cond_control = torch.stack([cond_control for _ in range(n_samples)], dim=0)
cond_control = einops.rearrange(cond_control, 'b h w c -> b c h w').clone()

seed = random.randint(0, 65535)
seed_everything(0)

cond = {"c_concat": [cond_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * n_samples)]}
un_cond = {"c_concat": None if guess_mode else [cond_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * n_samples)]}

shape = (4, h//8, w//8)

model.control_scales = [1 * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([1.0] * 13)

samples, inters = sampler.sample(ddim_steps, n_samples, shape, cond, verbose=False, eta=0.0, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=un_cond)


print("samples: ", samples.shape)
x_samples = model.decode_first_stage(samples)
x_samples = x_samples.squeeze(0)
x_samples = (x_samples + 1.0) / 2.0
x_samples = x_samples.transpose(0, 1).transpose(1, 2)
x_samples = x_samples.cpu().numpy()
x_samples = (x_samples * 255).astype(np.uint8)

image_name = output_img_path.split('/')[-1]
Image.fromarray(x_samples).save('../outputs/' + image_name)
