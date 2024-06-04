import torch
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from convertModels import savemodelDiffusers
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image, HWC3
# from annotator.hed import HEDdetector, nms
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
import cv2
import einops
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
import torchvision
import torchvision.transforms as transforms 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_name = "airplane"




@torch.no_grad()
def sample_model(model_name="CompVis/stable-diffusion-v1-4", sampler, size=512):

    image = load_image("test_input_images/airplane_sketch.png")
    hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
    erase_condition_image = hed(image, scribble=True)

    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda:0", torch.float16)

    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = preprocess(erase_condition_image).unsqueeze(0)

    # Encode the condition image to get the latent vector
    with torch.no_grad():
        latent_samples = vae.encode(image.to("cuda:0", torch.float16)).latent_dist.sample() * 0.18215

    return latent_samples



# model = load_model_from_config("./configs/controlnet/cldm_v15.yaml", "/workspace/control_sd15_scribble.pth", devices[0])
# sampler = DDIMSampler(model)

samples = sample_model(None, None, h=512, w=512, ddim_steps=50, n_samples=1)
print("got samples")






# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#                     prog = 'TrainESD',
#                     description = 'Finetuning stable diffusion model to erase concepts using ESD method')
#     parser.add_argument('--prompt', help='prompt corresponading to concept to erase', type=str, required=True)
#     parser.add_argument('--train_method', help='method of training', type=str, required=True)
#     parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
#     parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
#     parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=500)
#     parser.add_argument('--lr', help='learning rate used to train', type=int, required=False, default=1e-5)
#     parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='./configs/controlnet/cldm_v15.yaml')
#     # parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/controlnet_canny/control_sd15_canny.pth')
#     parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='/workspace/control_sd15_scribble.pth')
#     # parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
#     # parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
#     parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
#     parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
#     parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
#     parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
#     parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    
#     parser.add_argument('--erase_condition_image', help='Erase Condition Image', type=str, required=True, default="")
#     args = parser.parse_args()
    
#     prompt = args.prompt
#     train_method = args.train_method
#     start_guidance = args.start_guidance
#     negative_guidance = args.negative_guidance
#     iterations = args.iterations
#     lr = args.lr
#     config_path = args.config_path
#     ckpt_path = args.ckpt_path
#     diffusers_config_path = args.diffusers_config_path
#     devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
#     seperator = args.seperator
#     image_size = args.image_size
#     ddim_steps = args.ddim_steps
    
#     erase_condition_image = args.erase_condition_image

    # train_esd(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps)