import torch
from PIL import Image
import numpy as np
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector, nms
from diffusers.utils import load_image
import cv2
import einops
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DDPMScheduler
import torchvision.transforms as transforms
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import save_file
from tqdm import tqdm
import os
from pytorch_lightning import seed_everything
import random
import torch.nn.functional as F
import sys
import wandb 

def save_model_with_removal(path, params):
    if os.path.exists(path):
        os.remove(path)
    save_file(params, path)

# Variables to track the previous file paths
previous_unet_save_path = None
previous_cnet_save_path = None

wandb.login(key="6b9529ffc8d1630ecad71718647e2e14c98bf360")
wandb.init(project="sketch-erase")

class_name = sys.argv[1]
model_name = "runwayml/stable-diffusion-v1-5"
condition_image = f'test_input_images/{class_name}_sketch.png'
uncondition_image = "unconditional.png"
ddim_steps = 50
iterations = 1000
intermediate_model_dir = "intermediate_models_new"
controlnet_path = "converted_model"
save_interval = 50

# Initialize the noise scheduler
noise_scheduler = DDPMScheduler()

# Initialize HED detector
apply_hed = HEDdetector()

def check_tensor(tensor, name):
    assert not torch.isnan(tensor).any(), f'{name} contains NaNs'
    assert not torch.isinf(tensor).any(), f'{name} contains Infs'

def load_image_as_array(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        return np.array(img)

class LatentCaptureCallback:
    def __init__(self, target_step, timesteps_mapping):
        self.target_step = target_step
        self.target_timestep = timesteps_mapping[target_step - 1]
        self.captured_latents = None
        self.closest_timestep = None

    def __call__(self, step, timestep, latents):
        if timestep == self.target_timestep:
            self.captured_latents = latents.clone()
            self.closest_timestep = timestep

# Define the mapping from DDIM steps to timesteps
timesteps_mapping = [
    999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739,
    719, 699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 500, 480, 460,
    440, 420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180,
    160, 140, 120, 100, 80, 60, 40, 20
]

device = f'cuda:{sys.argv[2]}' 

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device, torch.float32)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def move_to_device(batch_encoding, device):
    return {key: tensor.to(device) if tensor.dtype == torch.int64 else tensor.to(device, torch.float32) for key, tensor in batch_encoding.items()}

def encode_image(image, vae):
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device, torch.float32)
    vae.eval()
    with torch.no_grad():
        latent_vector = vae.encode(image_tensor).latent_dist.sample() * 0.18215
    return latent_vector

def diffuse_to_random_timestep(latent_vector, timesteps, t, device):
    T = timesteps
    alpha = np.linspace(1, 0, T)
    sqrt_alpha = torch.tensor(np.sqrt(alpha), device=device, dtype=torch.float32)
    sqrt_one_minus_alpha = torch.tensor(np.sqrt(1 - alpha), device=device, dtype=torch.float32)
    noise = torch.randn_like(latent_vector)
    latent_noisy = sqrt_alpha[t] * latent_vector + sqrt_one_minus_alpha[t] * noise
    return latent_noisy

@torch.no_grad()
def sampler(pipeline, size, image_path, t):
    a_prompt = "best quality, extremely detailed"
    n_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, noisy"
    n_samples = 1

    # For the conditional image init
    image = load_image_as_array(image_path)
    input_image = HWC3(image)
    detected_map = apply_hed(resize_image(input_image, size))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, size)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    detected_map = nms(detected_map, 127, 3.0)
    detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
    detected_map[detected_map > 4] = 255
    detected_map[detected_map < 255] = 0

    control = torch.from_numpy(detected_map.copy()).float().to(device, torch.float32) / 255.0
    control = torch.stack([control for _ in range(n_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone().to(device, torch.float32)

    seed = random.randint(0, 65535)
    seed_everything(seed)

    # Prepare prompt and control images
    prompt = class_name + a_prompt
    negative_prompt = n_prompt
    
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    
    generator = torch.manual_seed(0)
    callback = LatentCaptureCallback(target_step=t, timesteps_mapping=timesteps_mapping)
    
    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control,
        num_inference_steps=50,
        guidance_scale=7.5,
        callback=callback,
        callback_steps=1,
        generator=generator,
        return_dict=True,
        )

    # Retrieve the captured latents
    captured_latents = callback.captured_latents

    return captured_latents.to(device, torch.float32)

def apply_unet_model(unet, vae, image_path, text_encoder, tokenizer, prompt, device, weight_dtype, noise_scheduler):
    # Load and preprocess the condition image
    image = load_image(image_path)
    image_tensor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])(image).unsqueeze(0).to(device, dtype=weight_dtype)

    # Encode the condition image to latents
    latents = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor

    # Sample noise to add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Tokenize the prompt for text conditioning
    text_inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    text_inputs = move_to_device(text_inputs, device)
    encoder_hidden_states = text_encoder(text_inputs['input_ids'])[0]

    # Predict the noise residual
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    return model_pred

controlnet_orig = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32, device=device, use_safetensors=True, safety_checker=None)
model_orig = StableDiffusionControlNetPipeline.from_pretrained(
    model_name, controlnet=controlnet_orig, torch_dtype=torch.float32, use_safetensors=True, device=device, safety_checker=None
)

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32, device=device, use_safetensors=True, safety_checker=None)
model = StableDiffusionControlNetPipeline.from_pretrained(
    model_name, controlnet=controlnet, use_safetensors=True, torch_dtype=torch.float32, device=device, safety_checker=None
)

# Freeze the original model parameters
for param in model_orig.controlnet.parameters():
    param.requires_grad = False

unet_train_method = 'xattn'

parameters = []
# Iterate through model parameters based on the training method
for name, param in model.unet.named_parameters():
    if unet_train_method == 'noxattn':
        if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
            print("noxattn")
        else:
            parameters.append(param)
    elif unet_train_method == 'selfattn':
        if 'attn1' in name:
            parameters.append(param)
    elif unet_train_method == 'xattn':
        if 'attn2' in name:
            parameters.append(param)
    elif unet_train_method == 'allattn':
        if 'attn1' in name or 'attn2' in name:
            parameters.append(param)
    elif unet_train_method == 'full':
        parameters.append(param)
    elif unet_train_method == 'notime':
        if not (name.startswith('out.') or 'time_embed' in name):
            parameters.append(param)
    elif unet_train_method == 'xlayer':
        if 'attn2' in name:
            if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                parameters.append(param)
    elif unet_train_method == 'selflayer':
        if 'attn1' in name:
            if 'input_blocks.4.' in name or 'input_blocks.7.' in name: 
                parameters.append(param)

controlnet_train_blocks = "attentions"
for name, param in model.controlnet.named_parameters():
    if controlnet_train_blocks in name:
        parameters.append(param)

print("controlnet parameters count: ", len(parameters))

model.to(device)

# Set the model to training mode
model.unet.train()
model.controlnet.train()

# Set up the optimizer and loss function
optimizer = torch.optim.Adam(parameters, lr=1e-5)
criteria = torch.nn.MSELoss()

def print_tensor_stats(tensor, name):
    print(f'{name} stats: min={tensor.min().item()}, max={tensor.max().item()}, mean={tensor.mean().item()}, std={tensor.std().item()}')

config = {
    "model_name": model_name,
    "ddim_steps": ddim_steps,
    "iterations": iterations,
    "unet_train_method": unet_train_method,
    "controlnet_train_blocks": controlnet_train_blocks,
    "learning_rate": 1e-5,
    "class_name": class_name
}

wandb.config.update(config)

pbar = tqdm(range(iterations))
for i in pbar:
    optimizer.zero_grad()

    t = torch.randint(0, ddim_steps, (1,), device=device, dtype=torch.long)

    model.to(device)
    model_orig.to(device)

    with torch.no_grad():
        z = sampler(model, 512, condition_image, t)

        a_prompt = "best quality, extremely detailed"
        unprompt = " " + a_prompt
        
        uncond_image_tensor = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])(load_image(uncondition_image)).unsqueeze(0).to(device, torch.float32)

        latent_uncondition_image = encode_image(load_image(condition_image), model_orig.vae.to(device, torch.float32)).detach()
        text_inputs = tokenizer(unprompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        text_inputs = move_to_device(text_inputs, device)
        text_embeddings = text_encoder(text_inputs["input_ids"]).last_hidden_state.detach()
        uncond = {'c_concat': [latent_uncondition_image], 'c_crossattn': [text_embeddings]}
        e_0 = apply_unet_model(model_orig.unet, model.vae, uncondition_image, text_encoder, tokenizer, unprompt, device, torch.float32, noise_scheduler)

        cprompt = class_name + a_prompt
        latent_condition_image = encode_image(load_image(condition_image), model_orig.vae.to(device, torch.float32)).detach()
        text_inputs = tokenizer(cprompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        text_inputs = move_to_device(text_inputs, device)
        text_embeddings = text_encoder(text_inputs['input_ids']).last_hidden_state.detach()
        cond = {'c_crossattn': [text_embeddings], 'c_concat': [latent_condition_image]}
        e_p = apply_unet_model(model_orig.unet, model.vae, condition_image, text_encoder, tokenizer, cprompt, device, torch.float32, noise_scheduler)

    latent_condition_image = encode_image(load_image(condition_image), model.vae.to(device, torch.float32)).detach()
    text_inputs = tokenizer(cprompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    text_inputs = move_to_device(text_inputs, device)
    text_embeddings = text_encoder(text_inputs['input_ids']).last_hidden_state.detach()
    cond = {'c_crossattn': [text_embeddings], 'c_concat': [latent_condition_image]}

    e_n = apply_unet_model(model.unet, model.vae, condition_image, text_encoder, tokenizer, cprompt, device, torch.float32, noise_scheduler)

    e_0 = e_0.detach()
    e_p = e_p.detach()

    eta = 0.5  # Example value, adjust as needed
    loss = criteria(e_n, e_0 - eta * (e_p - e_0))

    print("Loss: ", loss.item())
    wandb.log({"loss": loss.item()})

    loss.backward()
    pbar.set_postfix({"loss": loss.item()})
    optimizer.step()

    torch.cuda.empty_cache()
    
    if (i + 1) % save_interval == 0:
        # Define current file paths
        current_unet_save_path = os.path.join(f'{intermediate_model_dir}/{class_name}_unet_{unet_train_method}/', f'{class_name}_{unet_train_method}_{i+1}_unet.safetensors')
        current_cnet_save_path = os.path.join(f'{intermediate_model_dir}/{class_name}_cnet_{controlnet_train_blocks}/', f'{class_name}_{controlnet_train_blocks}_{i+1}_cnet.safetensors')

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(current_unet_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(current_cnet_save_path), exist_ok=True)

        # Remove previous files if they exist
        if previous_unet_save_path and os.path.exists(previous_unet_save_path):
            os.remove(previous_unet_save_path)
        if previous_cnet_save_path and os.path.exists(previous_cnet_save_path):
            os.remove(previous_cnet_save_path)

        # Save current model parameters
        unet_params = model.unet.state_dict()
        save_model_with_removal(current_unet_save_path, unet_params)
        print(f'Intermediate unet model saved at iteration {i+1} as {class_name}_{unet_train_method}_{i+1}_unet.safetensors')

        cnet_params = model.controlnet.state_dict()
        save_model_with_removal(current_cnet_save_path, cnet_params)
        print(f'Intermediate cnet model saved at iteration {i+1} as {class_name}_{controlnet_train_blocks}_{i+1}_cnet.safetensors')

        # Update previous file paths
        previous_unet_save_path = current_unet_save_path
        previous_cnet_save_path = current_cnet_save_path
