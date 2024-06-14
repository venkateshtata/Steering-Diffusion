import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

# Load the dataset from Hugging Face
dataset = load_dataset("cifar10", split="train")

# Preprocess the dataset
class HuggingFaceDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = self.dataset[idx]['img']
        text = "A photo of a " + self.dataset[idx]['label']
        inputs = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return {"inputs": inputs, "text": text}

# Initialize the model, scheduler, and processor
model_name = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(model_name)
pipeline.to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Prepare the dataset and dataloader
huggingface_dataset = HuggingFaceDataset(dataset, processor)
dataloader = DataLoader(huggingface_dataset, batch_size=4, shuffle=True)

# Initialize optimizer and scheduler
num_epochs = 5
optimizer = AdamW(pipeline.unet.parameters(), lr=1e-4)
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=500, num_training_steps=len(dataloader) * num_epochs)

# Initialize the noise scheduler
noise_scheduler = DDPMScheduler.from_config(model_name)

# Training loop
for epoch in range(num_epochs):
    pipeline.unet.train()
    for batch in tqdm(dataloader):
        # Get the input data and move it to the GPU
        inputs = batch['inputs'].to("cuda")
        texts = batch['text']
        
        # Tokenize the text
        text_inputs = tokenizer(texts, padding="max_length", return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
        text_inputs = text_inputs.input_ids.to("cuda")
        
        # Encode the text
        with torch.no_grad():
            encoder_hidden_states = text_encoder(text_inputs).last_hidden_state
        
        # Add noise to the inputs
        noise = torch.randn_like(inputs)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (inputs.shape[0],), device=inputs.device).long()
        noisy_inputs = noise_scheduler.add_noise(inputs, noise, timesteps)
        
        # Forward pass
        outputs = pipeline.unet(noisy_inputs, timesteps, encoder_hidden_states).sample
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(outputs, noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the model
# pipeline.save_pretrained("path/to/save/model")
