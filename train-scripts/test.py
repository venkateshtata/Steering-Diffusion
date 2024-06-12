import cv2
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Load image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png")
image = np.array(image)

# Apply Canny edge detection
low_threshold = 100
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

device = "cuda:0"

# Load ControlNet and pipeline
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32).to("cuda")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Enable model CPU offload
pipe.enable_model_cpu_offload()

# Seed for reproducibility
generator = torch.manual_seed(0)

# Tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device, torch.float32)

# Prompt
prompt = "a giant standing in a fantasy landscape, best quality"
text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
text_input_ids = text_inputs.input_ids.to(device)

# Get text embeddings
text_embeddings = text_encoder(text_input_ids)[0]

print("pipe.unet.sample_size: ", pipe.unet.sample_size)

# Generate initial noisy latents
latents = torch.randn((1, pipe.unet.in_channels, pipe.unet.sample_size, pipe.unet.sample_size), generator=generator).to(device, torch.float32)

# Define desired timestep
desired_timestep = 15

# Get scheduler's sigmas and timesteps
pipe.scheduler.set_timesteps(num_inference_steps=30)
sigmas = pipe.scheduler.sigmas.to(device, torch.float32)

# Add noise to the latents for the given timestep
noisy_latents = latents * sigmas[desired_timestep]

# Encode the canny image condition
control_image = torch.from_numpy(np.array(canny_image)).permute(2, 0, 1).unsqueeze(0).to(device, torch.float32) / 255.0

# Pass through ControlNet to get the conditioning output
with torch.no_grad():
    controlnet_output = pipe.controlnet(
        sample=noisy_latents.to(device, torch.float32),
        timestep=torch.tensor([desired_timestep]).to(device, torch.float32),
        encoder_hidden_states=text_embeddings.to(device, torch.float32),
        controlnet_cond=control_image.to(device, torch.float32),
        conditioning_scale=1.0,
        return_dict=True
    )
    
    print("controlnet_output shape: ", controlnet_output[1].shape)
    print("noisy_latents shape: ", noisy_latents.shape)

# Pass through UNet model to get noise prediction at the desired timestep
with torch.no_grad():
    noise_pred = pipe.unet(
        noisy_latents + controlnet_output,
        torch.tensor([desired_timestep]).to("cuda"),
        encoder_hidden_states=text_embeddings
    ).sample

# Convert noise prediction to numpy array for further processing if needed
noise_pred_np = noise_pred.cpu().numpy()
print("Noise prediction shape:", noise_pred_np.shape)

# Save the predicted noise as an image (optional visualization, this step may require normalization to [0, 255])
# Here we just save it directly for simplicity, but you may need to adjust it for proper visualization
noise_image = (noise_pred_np[0].transpose(1, 2, 0) * 255).astype(np.uint8)
Image.fromarray(noise_image).save("./noise_prediction.png")
