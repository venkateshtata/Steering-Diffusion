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

apply_hed = HEDdetector()

def load_image_as_array(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        image_array = np.array(img)
        return image_array


object_class = "cat"
input_type = "sketch"
model_used = "scribble"
unconditional_guidance_scale = 7
res = 512

# Configs
# resume_path = '/notebooks/Steering-Diffusion/models/control.pth' # default cehckpoint
resume_path = "/notebooks/Steering-Diffusion/models/test-compvis-word_airplane-method_full-sg_3-ng_1-iter_500-lr_1e-05_scribble/test-compvis-word_airplane-method_full-sg_3-ng_1-iter_500-lr_1e-05_scribble.pt"
N = 1
ddim_steps = 50


model = create_model('/notebooks/Steering-Diffusion/configs/controlnet/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cuda'), strict=False)
model = model.cuda()
sampler = DDIMSampler(model)
output_img_path = f'{object_class}-prompt_{object_class}-{input_type}_{unconditional_guidance_scale}-gs_{model_used}-full-with_uncond.png'


a_prompt = "best quality, extremely detailed"
n_prompt = "worst quality, low quality, noisy"
prompt = object_class
guess_mode = False
n_samples=1
h = res
w = res

low_threshold = 100
high_threshold = 200


# For the Conditional image init
image = load_image_as_array(f'/notebooks/Steering-Diffusion/test_input_images/{object_class}_{input_type}.png')

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
seed_everything(0)

cond = {"c_concat": [cond_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * n_samples)]}

un_cond = {"c_concat": [cond_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * n_samples)]}

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
Image.fromarray(x_samples).save("analysis_results/full_airplane_erased/"+image_name)
