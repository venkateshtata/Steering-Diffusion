
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






