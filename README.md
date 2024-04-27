
# Sketch Erase

## Requirements

```bash
pip install transformers==4.25.1 omegaconf einops matplotlib diffusers pytorch_lightning taming-transformers-rom1504 kornia clip open_clip_torch

export PYTHONPATH={project_root_directory}:$PYTHONPATH
```

## Training Command

```bash
python train-scripts/train-esd.py --train_method “xattn” --prompt="bird" --erase_condition_image="bird_canny.png"
```

## Test Script

```bash
python test_model.py
```

## Experiments and Results

### Experiment-1

#### Experiment Configuration:

- Training Iterations: **1000**
- Dataset used: Canny Edge
- DDIM Steps: **50**
- Learning Rate: **1e-5**
- Resolution: **512 x 512**
- Start Guidance: **3**
- Negative Guidance: **1**
- Unconditional Guidance Scale: **9.0**
- Positive Prompt: **"best quality, extremely detailed {Concept-Name}"**
- Negative Prompt: **"longbody, lowres,bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"**
- Layers Trained: **All**
- Concept To Erase: **Dog**

#### Experiment Results:

|           Prompt      | **"Bird"**        | **"Dog"**        |
|:--------------------------|:-------------------------:|:-------------------------:|
| Mask                 | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/test_images/bird_canny.png" width="128" height="128" alt="Mask for Bird"> | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/test_images/dog_canny.png" width="128" height="128" alt="Mask for Dog"> |
| **Default ControlNet**    | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/bird-prompt_bird-canny_output_default.png" width="128" height="128" alt="Default ControlNet Bird Output"> | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/dog-prompt_dog-canny_output_default.png" width="128" height="128" alt="Default ControlNet Dog Output"> |
| **Erased ControlNet**     | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/bird-prompt_bird-canny_output_v1.png" width="128" height="128" alt="Erased ControlNet Bird Output"> | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/dog-prompt_dog-canny_output_v1.png" width="128" height="128" alt="Erased ControlNet Dog Output"> |



### Experiment-2

#### Experiment Configuration:

- Training Iterations: **1000**
- Dataset used: Canny Edge
- DDIM Steps: **50**
- Learning Rate: **1e-5**
- Resolution: **512 x 512**
- Start Guidance: **3**
- Negative Guidance: **1**
- Unconditional Guidance Scale: **9.0**
- Positive Prompt: **"best quality, extremely detailed {Concept-Name}"**
- Negative Prompt: **"longbody, lowres,bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"**
- Layers Trained: **Self-Attention**
- Concept To Erase: **Dog**

#### Experiment Results:

|           Prompt      | **"Bird"**        | **"Dog"**        |
|:--------------------------|:-------------------------:|:-------------------------:|
| Mask                 | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/test_images/bird_canny.png" width="128" height="128" alt="Mask for Bird"> | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/test_images/dog_canny.png" width="128" height="128" alt="Mask for Dog"> |
| **Default ControlNet**    | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/bird-prompt_bird-canny_output_default.png" width="128" height="128" alt="Default ControlNet Bird Output"> | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/dog-prompt_dog-canny_output_default.png" width="128" height="128" alt="Default ControlNet Dog Output"> |
| **Erased ControlNet**     | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/bird-prompt_bird-canny_output_selfattn.png" width="128" height="128" alt="Erased ControlNet Bird Output"> | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/dog-prompt_dog-canny_output_selfattn.png" width="128" height="128" alt="Erased ControlNet Dog Output"> |


### Experiment-3

#### Experiment Configuration:

- Training Iterations: **1000**
- Dataset used: Canny Edge
- DDIM Steps: **50**
- Learning Rate: **1e-5**
- Resolution: **512 x 512**
- Start Guidance: **3**
- Negative Guidance: **1**
- Unconditional Guidance Scale: **9.0**
- Positive Prompt: **"best quality, extremely detailed {Concept-Name}"**
- Negative Prompt: **"longbody, lowres,bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"**
- Layers Trained: **Cross-Attention**
- Concept To Erase: **Dog**

#### Experiment Results:

|           Prompt      | **"Bird"**        | **"Dog"**        |
|:--------------------------|:-------------------------:|:-------------------------:|
| Mask                 | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/test_images/bird_canny.png" width="128" height="128" alt="Mask for Bird"> | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/test_images/dog_canny.png" width="128" height="128" alt="Mask for Dog"> |
| **Default ControlNet**    | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/bird-prompt_bird-canny_output_default.png" width="128" height="128" alt="Default ControlNet Bird Output"> | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/dog-prompt_dog-canny_output_default.png" width="128" height="128" alt="Default ControlNet Dog Output"> |
| **Erased ControlNet**     | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/bird-prompt_bird-canny_output_xattn.png" width="128" height="128" alt="Erased ControlNet Bird Output"> | <img src="https://github.com/venkateshtata/Steering-Diffusion/blob/main/outputs/dog-prompt_dog-canny_guide-9.0_output_xattn.png" width="128" height="128" alt="Erased ControlNet Dog Output"> |











