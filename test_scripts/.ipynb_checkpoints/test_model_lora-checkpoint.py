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

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel


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
resume_path = "/root/Steering-Diffusion/models/test-compvis-word_bird-method_full-sg_3-ng_1-iter_500-lr_1e-05_scribble/test-compvis-word_bird-method_full-sg_3-ng_1-iter_500-lr_1e-05_scribble.pt"
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

from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image


init_image = cond_img

# pipe = StableDiffusionImg2ImgPipeline.from_pretrained("/root/Steering-Diffusion/models/test-compvis-word_bird-method_full-sg_3-ng_1-iter_500-lr_1e-05_scribble/test-diffusers-word_bird-method_full-sg_3-ng_1-iter_500-lr_1e-05_scribble.pt", torch_dtype=torch.float16).to(
#     "cuda"
# )

prompt = "fish"
torch.manual_seed(1)


# controlnet = ControlNetModel.from_pretrained("/root/Steering-Diffusion/configs/controlnet/cldm_v15.yaml")
pipe = StableDiffusionControlNetPipeline.from_single_file("/root/Steering-Diffusion/models/test-compvis-word_bird-method_full-sg_3-ng_1-iter_500-lr_1e-05_scribble/test-diffusers-word_bird-method_full-sg_3-ng_1-iter_500-lr_1e-05_scribble.pt")


# image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]

print("loaded")