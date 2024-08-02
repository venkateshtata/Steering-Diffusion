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
from annotator.hed import HEDdetector, nms
import numpy as np
import cv2
import re

apply_hed = HEDdetector()

def load_image_as_array(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        image_array = np.array(img)
        return image_array

input_image_paths = ["test_input_images/airplane_sketch.png", "test_input_images/apple_sketch.png", "test_input_images/axe_sketch.png", "test_input_images/banana_sketch.png", "test_input_images/bicycle_sketch.png", "test_input_images/cat_sketch.png", "test_input_images/fish_sketch.png", "test_input_images/guitar_sketch.png", "test_input_images/mushroom_sketch.png"] 
unconditional_guidance_scale = 7
res = 512

# Configs
model_path = "/workspace/compvis-word_airplane-method_xattn-sg_3-ng_1-iter_500-lr_1e-05_scribble.pt"
N = 1
ddim_steps = 50

model = create_model('configs/controlnet/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(model_path, location='cuda'), strict=False)
model = model.cuda()
sampler = DDIMSampler(model)

a_prompt = "best quality, extremely detailed"
n_prompt = "worst quality, low quality, noisy"
guess_mode = False
n_samples=1
h = res
w = res

low_threshold = 100
high_threshold = 200

all_outputs = []

for input_image_path in input_image_paths:
    # For the Conditional image init
    image = load_image_as_array(input_image_path)

    prompt = input_image_path.split("/")[1].split("_")[0]

    input_image = HWC3(image)
    detected_map = apply_hed(resize_image(input_image, 512))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, 512)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    detected_map = nms(detected_map, 127, 3.0)
    detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
    detected_map[detected_map > 4] = 255
    detected_map[detected_map < 255] = 0

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(1)], dim=0)
    cond_control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    seed = random.randint(0, 65535)
    seed_everything(seed)

    cond = {"c_concat": [cond_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * n_samples)]}
    un_cond = {"c_concat": [cond_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * n_samples)]}
    shape = (4, h//8, w//8)
    model.control_scales = [1 * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([1.0] * 13)

    samples, inters = sampler.sample(ddim_steps, n_samples, shape, cond, verbose=False, eta=0.0, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=un_cond)

    x_samples = model.decode_first_stage(samples)
    x_samples = x_samples.squeeze(0)
    x_samples = (x_samples + 1.0) / 2.0
    x_samples = x_samples.transpose(0, 1).transpose(1, 2)
    x_samples = x_samples.cpu().numpy()
    x_samples = (x_samples * 255).astype(np.uint8)
    all_outputs.append(x_samples)

# Concatenate all images side by side
concatenated_image = np.concatenate(all_outputs, axis=1)
output_img_path = f'analysis_results/concatenated_output.png'
Image.fromarray(concatenated_image).save(output_img_path)

print(f"Saved concatenated image to {output_img_path}")
