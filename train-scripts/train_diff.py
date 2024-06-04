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
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import save_file, load_file



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda:0", torch.float16)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

model_name = "origin"
ddim_steps = 50
iterations = 500

def move_to_device(batch_encoding, device):
    return {key: tensor.to(device) if tensor.dtype == torch.int64 else tensor.to(device, torch.float16) for key, tensor in batch_encoding.items()}

def encode_image(image, vae):
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # Make sure the size is correct
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    print("vae.device: ", vae.device)
    image_tensor = preprocess(image).unsqueeze(0).to(vae.device, torch.float16)
    print("image tensor obtained")
    vae.eval()
    with torch.no_grad():

        latent_vector = vae.encode(image_tensor).latent_dist.sample() * 0.18215
    return latent_vector

def diffuse_to_random_timestep(latent_vector, timesteps, t, device):
    T = timesteps
    alpha = np.linspace(1, 0, T)
    sqrt_alpha = torch.tensor(np.sqrt(alpha), device=device, dtype=torch.float16)
    sqrt_one_minus_alpha = torch.tensor(np.sqrt(1 - alpha), device=device, dtype=torch.float16)
    
    # Perform the forward diffusion step to the random timestep
    noise = torch.randn_like(latent_vector)
    latent_noisy = sqrt_alpha[t] * latent_vector + sqrt_one_minus_alpha[t] * noise
    
    return latent_noisy

@torch.no_grad()
def sampler(model_name="runwayml/stable-diffusion-v1-5", size=512, image_path="test_input_images/airplane_sketch.png", t=1):

    image = load_image(image_path)
    hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
    erase_condition_image = hed(image, scribble=True)

    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda:0", torch.float16)

    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = preprocess(erase_condition_image).unsqueeze(0).to("cuda:0", torch.float16)

    # Encode the condition image to get the latent vector
    with torch.no_grad():
        latent_vector = vae.encode(image.to("cuda:0", torch.float16)).latent_dist.sample() * 0.18215

    latent_samples = diffuse_to_random_timestep(latent_vector, ddim_steps, t, device="cuda:0")

    return latent_samples


def apply_unet_model(unet, x_noisy, t, cond):
    """
    Apply the UNet model to predict the noise for a given noisy latent vector and timestep.
    
    Args:
        unet (torch.nn.Module): The UNet model.
        x_noisy (torch.Tensor): The noisy latent vector.
        t (torch.Tensor): The current timestep.
        cond (dict): Conditioning inputs, including 'c_crossattn' and 'c_concat'.

    Returns:
        torch.Tensor: The predicted noise.
    """
    assert isinstance(cond, dict)

    # Concatenate text conditioning inputs
    cond_txt = torch.cat(cond['c_crossattn'], 1)

    # Concatenate image conditioning inputs if available
    if cond['c_concat'] is not None:
        cond_img = torch.cat(cond['c_concat'], 1)
    else:
        cond_img = None

    # Apply the UNet model to predict the noise
    eps = unet(x_noisy, t, encoder_hidden_states=cond_txt, cross_attention_kwargs={"image_latents": cond_img}).sample

    return eps

model_name = "runwayml/stable-diffusion-v1-5"
generator = torch.manual_seed(0)
condition_image = "test_input_images/airplane_sketch.png"
uncondition_image = "unconditional.png"

# samples = sampler(model_name, 512)

class_name = "airplane"

controlnet_orig = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)

model_orig = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet_orig, torch_dtype=torch.float16,  use_safetensors=True
)

model = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,  use_safetensors=True
)

# Freeze the original model parameters
for param in model_orig.unet.parameters():
    param.requires_grad = False


# Define the train method
train_method = 'selfattn'  # Change this to your desired training method
parameters = []

# Iterate through model parameters based on the training method
for name, param in model.unet.named_parameters():
    if train_method == 'noxattn':
        if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
            print("name: ", name)
            
        else:
            parameters.append(param)
    elif train_method == 'selfattn':
        if 'attn1' in name:
            parameters.append(param)
    elif train_method == 'xattn':
        if 'attn2' in name:
            parameters.append(param)
    elif train_method == 'full':
        parameters.append(param)
    elif train_method == 'notime':
        if not (name.startswith('out.') or 'time_embed' in name):
            parameters.append(param)
    elif train_method == 'xlayer':
        if 'attn2' in name:
            if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                parameters.append(param)
    elif train_method == 'selflayer':
        if 'attn1' in name:
            if 'input_blocks.4.' in name or 'input_blocks.7.' in name: 
                print(name)
                parameters.append(param)

# Set the model to training mode
model.unet.train()

# Set up the optimizer and loss function
lr = 1e-4  # Learning rate
optimizer = torch.optim.Adam(parameters, lr=lr)
criteria = torch.nn.MSELoss()



pbar = tqdm(range(iterations))
for i in pbar:

        optimizer.zero_grad()

        t_enc = torch.randint(ddim_steps, (1,), device=device)
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=device)

        start_code = torch.randn((1, 4, 64, 64)).to(device)

        t = torch.randint(0, ddim_steps, (1,), device=device, dtype=torch.long)

        with torch.no_grad():

            
            z = sampler(model_name, 512, condition_image, t)

            # Get Unconditional scores from frozen model at timestep t and input z
            prompt = ""
            latent_uncondition_image = encode_image(load_image(uncondition_image), model_orig.vae.to("cuda:0", torch.float16))
            text_inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
            text_inputs = move_to_device(text_inputs, "cuda:0")
            text_embeddings = text_encoder(text_inputs['input_ids']).last_hidden_state

            uncond = {
                'c_crossattn': [text_embeddings],  # Replace with your actual text embeddings
                'c_concat': [latent_uncondition_image]  # Include the image latent vector as conditional input
            }
            e_0 = apply_unet_model(model_orig.unet.to("cuda:0", torch.float16), z.to("cuda:0", torch.float16), t.to("cuda:0", torch.float16), uncond)


            # Get Conditional scores from frozen model at timestep t and input z
            prompt = "airplane"
            latent_condition_image = encode_image(load_image(condition_image), model_orig.vae.to("cuda:0", torch.float16))
            text_inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
            text_inputs = move_to_device(text_inputs, "cuda:0")
            text_embeddings = text_encoder(text_inputs['input_ids']).last_hidden_state

            cond = {
                'c_crossattn': [text_embeddings],  # Replace with your actual text embeddings
                'c_concat': [latent_condition_image]  # Include the image latent vector as conditional input
            }
            e_p = apply_unet_model(model_orig.unet.to("cuda:0", torch.float16), z, t, cond)
        
        # Get Conditional scores from model at timestep t and input z
        prompt = "airplane"
        latent_condition_image = encode_image(load_image(condition_image), model.vae.to("cuda:0", torch.float16))
        text_inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        text_inputs = move_to_device(text_inputs, "cuda:0")
        text_embeddings = text_encoder(text_inputs['input_ids']).last_hidden_state

        cond = {
            'c_crossattn': [text_embeddings],  # Replace with your actual text embeddings
            'c_concat': [latent_condition_image]  # Include the image latent vector as conditional input
        }
        e_n = apply_unet_model(model.unet.to("cuda:0", torch.float16), z.to("cuda:0", torch.float16), t.to("cuda:0", torch.float16), cond)

        e_0.requires_grad = False
        e_p.requires_grad = False

        
        loss = criteria(e_n.to(device), e_0.to(device) - (1*(e_p.to(device) - e_0.to(device)))) #loss = criteria(e_n, e_0) works the best try 5000 epochs
        # update weights to erase the concept
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        optimizer.step()
        
torch.cuda.empty_cache()
        
import torch

# Save the ControlNet model parameters
model_save_path = "trained_model"
controlnet_save_path = os.path.join(model_save_path, "controlnet.safetensors")

# Extract ControlNet parameters and save them using safetensors
controlnet_params = model.controlnet.state_dict()
save_file(controlnet_params, controlnet_save_path)

print(f"ControlNet parameters saved to {controlnet_save_path}")

            

            
            

