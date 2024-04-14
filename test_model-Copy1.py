from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer

# Load the pipeline
pipeline = StableDiffusionPipeline.from_pretrained("./models/controlnet-word_bird-method_full-sg_3-ng_1-iter_500-lr_1e-05/controlnet-word_bird-method_full-sg_3-ng_1-iter_500-lr_1e-05.pt", torch_dtype=torch.float16).to("cuda")

# For deterministic results
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Load your conditional image
def load_image_as_array(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        image_array = np.array(img)
        return image_array

cond_img = load_image_as_array("./test_images/bird_canny.png")
cond_img = pipeline.numpy_to_pil(cond_img)  # Convert numpy image to PIL for the pipeline

# Generate images
prompt = "A detailed, high-quality painting of a bird"
output_images = pipeline(prompt=prompt, init_image=cond_img, num_inference_steps=50, strength=0.75)

# Save the output
output_images["sample"][0].save("output_bird.png")
