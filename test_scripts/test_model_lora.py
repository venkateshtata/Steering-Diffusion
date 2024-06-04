from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
import einops

from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector, nms
import numpy as np
import cv2

class_name = "airplane"

apply_hed = HEDdetector()

def load_image_as_array(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        image_array = np.array(img)
        return image_array

image = load_image_as_array(f'/notebooks/Steering-Diffusion/test_input_images/{class_name}_sketch.png')


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
control = einops.rearrange(control, 'b h w c -> b c h w').clone()




controlnet = ControlNetModel.from_pretrained("/notebooks/Steering-Diffusion/models/diff_models/test-compvis-word_airplane-method_notime-sg_3-ng_1-iter_500-lr_1e-05_scribble/")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", controlnet=controlnet, safety_checker=None, guess_mode=False
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

negative_prompt = "worst quality, low quality, noisy"




image = pipe(class_name, control, num_inference_steps=50, guidance_scale=7, negative_prompt=negative_prompt).images[0]

image.save(f'analysis_results_lora/{class_name}_scribble_defaultmodel_no_uncond.png')