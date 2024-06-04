from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
from diffusers.utils import load_image

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

class_name = "airplane"

image = load_image(f'/notebooks/Steering-Diffusion/test_input_images/{class_name}_sketch.png')

image = hed(image, scribble=True)

# controlnet = ControlNetModel.from_pretrained(
#     "/notebooks/Steering-Diffusion/converted_model/", torch_dtype=torch.float16, local_files_only=True
# )

# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16, local_files_only=True
# )

# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# # Remove if you do not have xformers installed
# # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# # for installation instructions
# # pipe.enable_xformers_memory_efficient_attention()

# pipe.enable_model_cpu_offload()

# image = pipe(class_name, image, num_inference_steps=50).images[0]

# image.save(f'{class_name}_scribble_out.png')


from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image

# download an image
# image = load_image(
# f'/notebooks/Steering-Diffusion/test_input_images/{class_name}_sketch.png'
# )
# np_image = np.array(image)

# # get canny image
# np_image = cv2.Canny(np_image, 100, 200)
# np_image = np_image[:, :, None]
# np_image = np.concatenate([np_image, np_image, np_image], axis=2)
# canny_image = Image.fromarray(np_image)

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("/notebooks/Steering-Diffusion/converted_model/", safety_checker=None)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
"CompVis/stable-diffusion-v1-4", controlnet=controlnet, safety_checker=None
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# generate image
generator = torch.manual_seed(0)
image = pipe(
"airplane",
num_inference_steps=50,
generator=generator,
image=image,
control_image=image,
).images[0]

image.save(f'{class_name}_scribble_out.png')
