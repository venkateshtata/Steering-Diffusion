from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
# from controlnet_aux import HEDdetector
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from annotator.hed import HEDdetector, nms
from annotator.util import HWC3
from annotator.util import resize_image
import einops
from safetensors.torch import load_file as load_safetensors

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
cond_control = einops.rearrange(control, 'b h w c -> b c h w').clone()


# Load ControlNet model and its components separately
controlnet = ControlNetModel.from_pretrained("/notebooks/Steering-Diffusion/converted_model/", safety_checker=None)

controlnet_state_dict = load_safetensors("/notebooks/Steering-Diffusion/trained_model/controlnet.safetensors")
controlnet.load_state_dict(controlnet_state_dict)


# vae = torch.load("/notebooks/Steering-Diffusion/converted_model/vae.pth")

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
# Load the Stable Diffusion pipeline with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    vae=vae,
    controlnet=controlnet,
    safety_checker=None
)

print("pipe keys: ", pipe)

# Assign the loaded VAE and UNet to the pipeline
# unet_state_dict = load_safetensors("/notebooks/Steering-Diffusion/trained_model/unet.safetensors")
# pipe.unet.load_state_dict(unet_state_dict)

# Speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Generate image
generator = torch.manual_seed(0)
output_image = pipe(
    class_name,
    num_inference_steps=50,
    generator=generator,
    image=cond_control,
).images[0]

# Save output image
output_image.save(f'{class_name}_scribble_out_defaultmodel.png')