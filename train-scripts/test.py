import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image

model_name = "CompVis/stable-diffusion-v1-4"

# Load the ControlNet pipeline
controlnet = ControlNetModel.from_pretrained("/notebooks/Steering-Diffusion/converted_model", torch_dtype=torch.float32, device="cuda:0", use_safetensors=True, safety_checker=None)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    model_name, controlnet=controlnet, use_safetensors=True, torch_dtype=torch.float32, device="cuda:0", safety_checker=None
)

pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

pipeline = pipeline.to("cuda")

# Define the callback function to capture latents at a specific timestep
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

# Define the target DDIM step (1-50) for which we want to capture the latents
target_step = 10
callback = LatentCaptureCallback(target_step=target_step, timesteps_mapping=timesteps_mapping)

# Load an example image
image = Image.open("/notebooks/Steering-Diffusion/test_input_images/fish_sketch.png")

# Define the prompt
prompt = "fish"

# Run the pipeline with the callback to capture latents
generator = torch.Generator("cuda").manual_seed(42)
output = pipeline(
    prompt=prompt,
    image=image,
    num_inference_steps=50,
    guidance_scale=7.5,
    callback=callback,
    callback_steps=1,
    generator=generator,
    return_dict=True,
)

# Retrieve the captured latents
captured_latents = callback.captured_latents
closest_timestep = callback.closest_timestep

# Print the shape of the captured latents to verify
if captured_latents is not None:
    print(f"Captured latents at timestep {closest_timestep}: {captured_latents.shape}")
else:
    print(f"No latents captured at timestep corresponding to step {target_step}.")
