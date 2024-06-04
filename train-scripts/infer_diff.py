import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
import torchvision.transforms as transforms
from diffusers.utils import load_image
from safetensors.torch import load_file
import os

# Function to move batch encoding to device
def move_to_device(batch_encoding, device):
    return {key: tensor.to(device) if tensor.dtype == torch.int64 else tensor.to(device, torch.float16) for key, tensor in batch_encoding.items()}

# Function to encode image
def encode_image(image, vae):
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # Make sure the size is correct
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(vae.device, torch.float16)
    vae.eval()
    with torch.no_grad():
        latent_vector = vae.encode(image_tensor).latent_dist.sample() * 0.18215
    return latent_vector

# Load the ControlNet parameters from safetensors file
model_save_path = "/root/Steering-Diffusion/trained_model/"
controlnet_save_path = os.path.join(model_save_path, "controlnet.safetensors")
controlnet_params = load_file(controlnet_save_path)

# Create and load the ControlNet model
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
controlnet.load_state_dict(controlnet_params)
controlnet.to("cuda:0")

# Load the Stable Diffusion ControlNet Pipeline
model = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
).to("cuda:0")

# Load the VAE and tokenizer
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to("cuda:0", torch.float16)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda:0", torch.float16)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Define the function to perform inference
def inference(prompt, condition_image_path):
    # Load and encode the condition image
    condition_image = load_image(condition_image_path)
    latent_condition_image = encode_image(condition_image, vae)

    # Tokenize and encode the prompt
    text_inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    text_inputs = move_to_device(text_inputs, "cuda:0")
    text_embeddings = text_encoder(text_inputs['input_ids']).last_hidden_state

    # Create conditioning dictionary
    cond = {
        'c_crossattn': [text_embeddings],  # Replace with your actual text embeddings
        'c_concat': [latent_condition_image]  # Include the image latent vector as conditional input
    }

    # Generate the image using the model
    generated_images = model(prompt, image=condition_image, guidance_scale=7.5, num_inference_steps=50).images
    return generated_images

# Perform inference
prompt = "airplane"
condition_image_path = "test_input_images/airplane_sketch.png"
generated_images = inference(prompt, condition_image_path)

# Save or display the generated images
for idx, image in enumerate(generated_images):
    image.save(f"generated_image_{idx}.png")