import torch
from PIL import Image
import numpy as np
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector, nms
from diffusers.utils import load_image
import cv2
import einops
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
import torchvision.transforms as transforms
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import save_file
from tqdm import tqdm
import os
from pytorch_lightning import seed_everything
import random

# Initialize HED detector
apply_hed = HEDdetector()

def check_tensor(tensor, name):
    assert not torch.isnan(tensor).any(), f'{name} contains NaNs'
    assert not torch.isinf(tensor).any(), f'{name} contains Infs'

def load_image_as_array(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        return np.array(img)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device, torch.float32)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

ddim_steps = 50
iterations = 500

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
    control = einops.rearrange(control, 'b h w c -> b c h w').clone().to("cuda:0", torch.float32)

    seed = random.randint(0, 65535)
    seed_everything(seed)

    # Prepare prompt and control images
    prompt = "fish " + a_prompt
    negative_prompt = n_prompt
    
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    # pipeline.enable_model_cpu_offload()

    # Generate images
    generator = torch.manual_seed(0)
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=ddim_steps,
        guidance_scale=1,
        image=control,
        generator=generator
    ).images[0]
    
    latent_vector = encode_image(load_image(image), pipeline.vae.to(device, torch.float32))
    
    noisy_latent_vector = diffuse_to_random_timestep(latent_vector, 50, t, device)
    
    return noisy_latent_vector.to(device, torch.float32)

def apply_unet_model(unet, x_noisy, t, cond):
    cond_txt = torch.cat(cond['c_crossattn'], 1)
    
    # Check inputs for NaNs
    check_tensor(x_noisy, "x_noisy in apply_unet_model")
    check_tensor(t, "t in apply_unet_model")
    check_tensor(cond_txt, "cond_txt in apply_unet_model")
    
    # Forward pass through UNet
    eps = unet(x_noisy, t, encoder_hidden_states=cond_txt).sample

    # Check for NaNs and print tensor stats
    check_tensor(eps, "eps in apply_unet_model")
    print_tensor_stats(eps, "eps in apply_unet_model")
    
    return eps


class_name = "fish"
model_name = "CompVis/stable-diffusion-v1-4"
condition_image = f'test_input_images/{class_name}_sketch.png'
uncondition_image = "unconditional.png"

controlnet_orig = ControlNetModel.from_pretrained("/notebooks/Steering-Diffusion/converted_model", torch_dtype=torch.float32, device="cuda:0", use_safetensors=True, safety_checker = None)
model_orig = StableDiffusionControlNetPipeline.from_pretrained(
    model_name, controlnet=controlnet_orig, torch_dtype=torch.float32, use_safetensors=True, device="cuda:0", safety_checker = None
)

controlnet = ControlNetModel.from_pretrained("/notebooks/Steering-Diffusion/converted_model", torch_dtype=torch.float32, device="cuda:0", use_safetensors=True, safety_checker = None)
model = StableDiffusionControlNetPipeline.from_pretrained(
    model_name, controlnet=controlnet, use_safetensors=True, torch_dtype=torch.float32, device="cuda:0", safety_checker = None
)

# Freeze the original model parameters
for param in model_orig.controlnet.parameters():
    param.requires_grad = False

# Define the train method
train_method = 'xattn'

parameters = []
# Iterate through model parameters based on the training method
for name, param in model.unet.named_parameters():
    if train_method == 'noxattn':
        if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
            print("noxattn")
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
                # print(name)
                parameters.append(param)

model.to(device)
# Set the model to training mode
model.unet.train()

torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)

# Set up the optimizer and loss function
optimizer = torch.optim.Adam(parameters, lr=1e-5)
criteria = torch.nn.MSELoss()

def print_tensor_stats(tensor, name):
    print(f'{name} stats: min={tensor.min().item()}, max={tensor.max().item()}, mean={tensor.mean().item()}, std={tensor.std().item()}')

pbar = tqdm(range(iterations))
for _ in pbar:
    optimizer.zero_grad()

    t_enc = torch.randint(ddim_steps, (1,), device=device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)

    t = torch.randint(0, ddim_steps, (1,), device=device, dtype=torch.long)

    model.to("cuda:0")
    model_orig.to("cuda:0")

    with torch.no_grad():
        z = sampler(model, 512, condition_image, t)
        check_tensor(z, "z")
        print_tensor_stats(z, "z")

        a_prompt = "best quality, extremely detailed"

        unprompt = " " + a_prompt
        latent_uncondition_image = encode_image(load_image(uncondition_image), model.vae.to(device, torch.float32)).detach()
        check_tensor(latent_uncondition_image, "latent_uncondition_image")
        print_tensor_stats(latent_uncondition_image, "latent_uncondition_image")
        text_inputs = tokenizer(unprompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        text_inputs = move_to_device(text_inputs, device)
        text_embeddings = text_encoder(text_inputs["input_ids"]).last_hidden_state.detach()
        check_tensor(text_embeddings, "text_embeddings")
        print_tensor_stats(text_embeddings, "text_embeddings")
        uncond = {'c_concat': [latent_uncondition_image], 'c_crossattn': [text_embeddings]}
        e_0 = apply_unet_model(model_orig.unet.to(device, torch.float32), z.to(device, torch.float32), t.to(device, torch.float32), uncond)
        check_tensor(e_0, "e_0")
        print_tensor_stats(e_0, "e_0")

        cprompt = "fish " + a_prompt
        latent_condition_image = encode_image(load_image(condition_image), model.vae.to(device, torch.float32)).detach()
        check_tensor(latent_condition_image, "latent_condition_image")
        print_tensor_stats(latent_condition_image, "latent_condition_image")
        text_inputs = tokenizer(cprompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        text_inputs = move_to_device(text_inputs, device)
        text_embeddings = text_encoder(text_inputs['input_ids']).last_hidden_state.detach()
        check_tensor(text_embeddings, "text_embeddings")
        print_tensor_stats(text_embeddings, "text_embeddings")
        cond = {'c_crossattn': [text_embeddings], 'c_concat': [latent_condition_image]}
        e_p = apply_unet_model(model_orig.unet.to(device, torch.float32), z.to(device, torch.float32), t.to(device, torch.float32), cond)
        check_tensor(e_p, "e_p")
        print_tensor_stats(e_p, "e_p")

    latent_condition_image = encode_image(load_image(condition_image), model.vae.to(device, torch.float32)).detach()
    check_tensor(latent_condition_image, "latent_condition_image")
    print_tensor_stats(latent_condition_image, "latent_condition_image")
    text_inputs = tokenizer(cprompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    text_inputs = move_to_device(text_inputs, device)
    text_embeddings = text_encoder(text_inputs['input_ids']).last_hidden_state.detach()
    cond = {'c_crossattn': [text_embeddings], 'c_concat': [latent_condition_image]}
    check_tensor(latent_condition_image, "latent_condition_image")
    check_tensor(text_embeddings, "text_embeddings")

    # Clamp and check for NaNs in e_n
    e_n = apply_unet_model(model.unet.to(device, torch.float32), z.to(device, torch.float32), t.to(device, torch.float32), cond)
    e_n = torch.clamp(e_n, -1, 1)
    check_tensor(e_n, "e_n")
    print_tensor_stats(e_n, "e_n")

    e_0 = e_0.detach()
    e_p = e_p.detach()

    loss = criteria(e_n, e_0 - (1 * (e_p - e_0)))
    check_tensor(loss, "loss")

    print("Loss: ", loss.item())

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
    pbar.set_postfix({"loss": loss.item()})
    optimizer.step()

    torch.cuda.empty_cache()

# Save the trained model
model_save_path = "trained_model"
os.makedirs(model_save_path, exist_ok=True)
unet_save_path = os.path.join(model_save_path, f'{class_name}_{train_method}_{iterations}_unet.safetensors')

unet_params = model.unet.state_dict()
save_file(unet_params, unet_save_path)

print(f'model saved as {class_name}_{train_method}_{iterations}_unet.safetensors')
