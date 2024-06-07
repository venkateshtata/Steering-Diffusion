from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler, ControlNetModel
import torch
from diffusers.utils import load_image
from diffusers import AutoencoderKL
import numpy as np
import cv2
from annotator.hed import HEDdetector, nms
from annotator.util import HWC3, resize_image
import einops
from safetensors.torch import load_file as load_safetensors

class_name = "fish"

unet_model_path = "/notebooks/Steering-Diffusion/intermediate_models/fish_xattn_400_unet.safetensors"
controlnet_model_path = "/notebooks/Steering-Diffusion/intermediate_models/fish_xattn_400_cnet.safetensors"

iterations = unet_model_path.split(".")[0].split("_")[-2]
train_method = unet_model_path.split(".")[0].split("_")[-3]
erased_class = unet_model_path.split(".")[0].split("/")[-1].split("_")[0]

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

# Load ControlNet model
controlnet = ControlNetModel.from_pretrained("/notebooks/Steering-Diffusion/converted_model", torch_dtype=torch.float32, device="cuda:0", use_safetensors=True, safety_checker = None)

unet_state_dict = load_safetensors(unet_model_path)
controlnet_state_dict = load_safetensors(controlnet_model_path)

# Load the Stable Diffusion pipeline with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    controlnet=controlnet,
    safety_checker=None
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_state_dict(unet_state_dict, strict=False)
pipe.controlnet.load_state_dict(controlnet_state_dict, strict=False)


# Load the VAE model
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", safety_checker=None)
pipe.vae = vae

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
output_image.save(f'testing_outputs/{class_name}-{erased_class}-erased_{train_method}_{iterations}.png')
print(f'Output saved to testing_outputs/{class_name}-{erased_class}-erased_{train_method}_{iterations}.png')
