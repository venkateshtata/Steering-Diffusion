from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ldm.util import instantiate_from_config
import random
import glob
import re
import shutil
import pdb
import argparse
from convertModels import savemodelDiffusers
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image, HWC3
import einops
from pytorch_lightning import seed_everything
from annotator.canny import CannyDetector

apply_canny = CannyDetector()


def load_image_as_array(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        image_array = np.array(img)
        return image_array


# Util Functions

def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    batch_size = 4
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False
    model = create_model(config).to(device)
    model.load_state_dict(load_state_dict(ckpt, location='cuda'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.eval()
    model.cond_stage_model.device = device
    return model


@torch.no_grad()
def sample_model(model, sampler, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    a_prompt = "best quality, extremely detailed"
    n_prompt = "longbody, lowres,bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    # prompt = prompt
    guess_mode = False
    low_threshold = 100
    high_threshold = 200
    n_samples = 1

    # For the conditional image init
    cond_img = load_image_as_array(erase_condition_image)
    cond_img = resize_image(HWC3(cond_img), 512)
    h, w = 512, 512
    
    # cond_detected_map = np.zeros_like(cond_img, dtype=np.uint8)
    # cond_detected_map[np.min(cond_img, axis=2) < 127] = 255
    cond_detected_map = apply_canny(cond_img, low_threshold, high_threshold)
    cond_detected_map = HWC3(cond_detected_map)

    cond_control = torch.from_numpy(cond_detected_map.copy()).float().cuda() / 255.0
    cond_control = torch.stack([cond_control for _ in range(n_samples)], dim=0)
    cond_control = einops.rearrange(cond_control, 'b h w c -> b c h w').clone()

    seed = random.randint(0, 65535)
    seed_everything(0)

    cond = {"c_concat": [cond_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * n_samples)]}
    un_cond = {"c_concat": None if guess_mode else [cond_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * n_samples)]}

    shape = (4, h//8, w//8)

    model.control_scales = [1 * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([1.0] * 13)
    
    samples_ddim, inters = sampler.sample(ddim_steps, n_samples, shape, cond, verbose=False, eta=0.0, unconditional_guidance_scale=9.0, unconditional_conditioning=un_cond)

    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)


def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler


def train_esd(prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, diffusers_config_path, devices, seperator=None, image_size=512, ddim_steps=50):

    # PROMPT CLEANING
    word_print = prompt.replace(' ','')
    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    print(words)
    ddim_eta = 0

    model_orig, sampler_orig, model, sampler = get_models(config_path, ckpt_path, devices)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == 'full':
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                print(name)
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    print(name)
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    print(name)
                    parameters.append(param)
    # set model to train
    model.train()
    
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda s, code, t: sample_model(model, sampler,
                                                                 image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    name = f'compvis-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}'
    # TRAINING CODE
    pbar = tqdm(range(iterations))
    for i in pbar:
        word = random.sample(words,1)[0]

        opt.zero_grad()

        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            # generate an image with the concept from ESD model
            z = quick_sample_till_t(start_guidance, start_code, int(t_enc)) # emb_p seems to work better instead of emb_0
            # print("shape of z: ", z.shape)
            
            #Get outputs from frozen model
            a_prompt = "best quality, extremely detailed"
            n_prompt = "longbody, lowres,bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
            guess_mode = False
            n_samples=1
            low_threshold = 100
            high_threshold = 200
            
            # get Un-conditional scores from frozen model at time step t and image z
            unprompt = ""
            uncond_img = load_image_as_array("unconditional.png")
            uncond_img = resize_image(HWC3(uncond_img), 512)
            
            h, w = 512, 512

            # uncond_detected_map = np.zeros_like(uncond_img, dtype=np.uint8)
            # uncond_detected_map[np.min(uncond_img, axis=2) < 127] = 255
            uncond_detected_map = apply_canny(uncond_img, low_threshold, high_threshold)
            uncond_detected_map = HWC3(uncond_detected_map)

            uncond_control = torch.from_numpy(uncond_detected_map.copy()).float().cuda() / 255.0
            uncond_control = torch.stack([uncond_control for _ in range(n_samples)], dim=0)
            uncond_control = einops.rearrange(uncond_control, 'b h w c -> b c h w').clone()

            uncond = {"c_concat": [uncond_control], "c_crossattn": [model_orig.get_learned_conditioning([unprompt + ', ' + a_prompt] * n_samples)]}
            model_orig.control_scales = [1 * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([1.0] * 13)
            e_0 = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), uncond)
            # print("shape of e_0: ", e_0.shape)
            
            # get Conditional scores from frozen model at time step t and image z
            cprompt = prompt
            cond_img = load_image_as_array(erase_condition_image)
            cond_img = resize_image(HWC3(cond_img), 512)
            
            h, w = 512, 512

            # cond_detected_map = np.zeros_like(cond_img, dtype=np.uint8)
            # cond_detected_map[np.min(cond_img, axis=2) < 127] = 255
            cond_detected_map = apply_canny(cond_img, low_threshold, high_threshold)
            cond_detected_map = HWC3(cond_detected_map)

            cond_control = torch.from_numpy(cond_detected_map.copy()).float().cuda() / 255.0
            cond_control = torch.stack([cond_control for _ in range(n_samples)], dim=0)
            cond_control = einops.rearrange(cond_control, 'b h w c -> b c h w').clone()

            cond = {"c_concat": [cond_control], "c_crossattn": [model_orig.get_learned_conditioning([cprompt + ', ' + a_prompt] * 1)]}
            model_orig.control_scales = [1 * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([1.0] * 13)
            e_p = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), cond)
            # print("shape of e_p: ", e_p.shape)
            
        # breakpoint()
        # get conditional score from ESD model
        e_prompt = prompt
        e_cond_img = load_image_as_array(erase_condition_image)
        e_cond_img = resize_image(HWC3(e_cond_img), 512)
        
        h, w = 512, 512

        # e_cond_detected_map = np.zeros_like(e_cond_img, dtype=np.uint8)
        # e_cond_detected_map[np.min(e_cond_img, axis=2) < 127] = 255
        e_cond_detected_map = apply_canny(e_cond_img, low_threshold, high_threshold)
        e_cond_detected_map = HWC3(e_cond_detected_map)

        e_cond_control = torch.from_numpy(e_cond_detected_map.copy()).float().cuda() / 255.0
        e_cond_control = torch.stack([e_cond_control for _ in range(n_samples)], dim=0)
        e_cond_control = einops.rearrange(e_cond_control, 'b h w c -> b c h w').clone()

        e_cond = {"c_concat": [e_cond_control], "c_crossattn": [model.get_learned_conditioning([e_prompt + ', ' + a_prompt] * 1)]}
        model.control_scales = [1 * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([1.0] * 13)
        e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), e_cond)
        # print("shape of e_n: ", e_n.shape)
        
        e_0.requires_grad = False
        e_p.requires_grad = False
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (negative_guidance*(e_p.to(devices[0]) - e_0.to(devices[0])))) #loss = criteria(e_n, e_0) works the best try 5000 epochs
        # update weights to erase the concept
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()
        # save checkpoint and loss curve
        if (i+1) % 500 == 0 and i+1 != iterations and i+1>= 500:
            save_model(model, name, i-1, save_compvis=True, save_diffusers=False)

        if i % 100 == 0:
            save_history(losses, name, word_print)

    model.eval()

    save_model(model, name, None, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    save_history(losses, name, word_print)

def save_model(model, name, num, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    # SAVE MODEL

#     PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'

    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/{name}.pt'
    if save_compvis:
        print("saving compvis format")
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, device=device )

def save_history(losses, name, word_print):
    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--prompt', help='prompt corresponading to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=500)
    parser.add_argument('--lr', help='learning rate used to train', type=int, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/controlnet/cldm_v15.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/controlnet_canny/control_sd15_canny.pth')
    # parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    # parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    
    parser.add_argument('--erase_condition_image', help='Erase Condition Image', type=str, required=True, default="")
    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    
    erase_condition_image = args.erase_condition_image

    train_esd(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps)