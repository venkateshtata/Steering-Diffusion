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
import sys

erase_class_name = sys.argv[1]

test_class_name = sys.argv[2]

unet_model_path = f'intermediate_models/{erase_class_name}_unet_xattn/{erase_class_name}_xattn_100_unet.safetensors'
controlnet_model_path = f'intermediate_models/{erase_class_name}_cnet_xattn/{erase_class_name}_xattn_100_cnet.safetensors'

iterations = unet_model_path.split(".")[0].split("_")[-2]
unet_train_method = unet_model_path.split(".")[0].split("_")[-3]
cnet_train_method = controlnet_model_path.split(".")[0].split("_")[-3]
erased_class = unet_model_path.split(".")[0].split("/")[-1].split("_")[0]

apply_hed = HEDdetector()

def load_image_as_array(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        image_array = np.array(img)
        return image_array

image = load_image_as_array(f'test_input_images/{test_class_name}_sketch.png')

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
controlnet = ControlNetModel.from_pretrained("converted_model", torch_dtype=torch.float32, device="cuda:0", use_safetensors=True, safety_checker = None)

unet_state_dict = load_safetensors(unet_model_path)
controlnet_state_dict = load_safetensors(controlnet_model_path)

# Load the Stable Diffusion pipeline with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_state_dict(unet_state_dict, strict=False)
pipe.controlnet.load_state_dict(controlnet_state_dict, strict=False)


# Load the VAE model
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", safety_checker=None)
pipe.vae = vae

# Speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Generate image
generator = torch.manual_seed(0)
output_image = pipe(
    test_class_name,
    num_inference_steps=50,
    generator=generator,
    image=cond_control,
).images[0]

# Save output image
output_image.save(f'outputs/{erased_class}-erased_{test_class_name}_{unet_train_method}-unet_{cnet_train_method}-cnet_{iterations}.png')
print(f'Output saved to outputs/{erased_class}-erased_{test_class_name}_{unet_train_method}-unet_{cnet_train_method}-cnet_{iterations}.png')
