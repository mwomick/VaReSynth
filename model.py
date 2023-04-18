import torch
import torch.nn as nn
import math
from typing import Optional
from diffusers import AutoencoderKL
from unet import SpatialDiffusionUNet

from transformers import CLIPTextModel, CLIPTokenizer


class VaReSynth(nn.Module):
    def __init__(self, hf_key=None, sd_version='2.0'):
        super().__init__()

        self.main_device = torch.device('cuda:0')
        if torch.device('cuda:1'):
            self.bg_device = torch.device('cuda:1')
        else:
            self.bg_device = self.main_device

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.bg_device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.bg_device) # out = [77, 1024]
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.unet = SpatialDiffusionUNet(sample_size=64,
                                         in_channels=4,
                                         out_channels=4,
                                         block_out_channels=(320, 640, 1280, 1280),
                                         cross_attention_dim=1024,
                                         attention_head_dim=[5, 10, 20, 20],
                                         )
        

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    
    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs


    @torch.no_grad()
    def encode_image(self, image):
        return self.vae.encode(image).latent_dist


    def forward(self, latents, log_snrs, raw_prompts, pos):
        text_cond = self.get_text_embeds(raw_prompts)
        return self.unet(latents, log_snrs, text_cond, pos)
