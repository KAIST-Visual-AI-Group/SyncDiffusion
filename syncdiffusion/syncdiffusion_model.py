import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.autograd import grad
import argparse
from tqdm import tqdm

from syncdiffusion.utils import *
import lpips
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

class SyncDiffusion(nn.Module):
    def __init__(self, device='cuda', sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Load pretrained models from HuggingFace
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        
        # Freeze models
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.unet.eval() 
        self.vae.eval()
        self.text_encoder.eval()
        print(f'[INFO] loaded stable diffusion!')

        # Set DDIM scheduler
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        # load perceptual loss (LPIPS)
        self.percept_loss = lpips.LPIPS(net='vgg').to(self.device)
        print(f'[INFO] loaded perceptual loss!')

    def get_text_embeds(self, prompt, negative_prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Repeat for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Concatenate for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
    
    def sample_syncdiffusion(
        self, 
        prompts, 
        negative_prompts="", 
        height=512, 
        width=2048, 
        latent_size=64,                     # fix latent size to 64 for Stable Diffusion
        num_inference_steps=50,
        guidance_scale=7.5, 
        sync_weight=20,                     # gradient descent weight 'w' in the paper
        sync_freq=1,                        # sync_freq=n: perform gradient descent every n steps
        sync_thres=50,                      # sync_thres=n: compute SyncDiffusion only for the first n steps
        sync_decay_rate=0.95,               # decay rate for sync_weight, set as 0.95 in the paper        
        stride=16,                          # stride for latents, set as 16 in the paper           
    ):  
        assert height >= 512 and width >= 512, 'height and width must be at least 512'
        assert height % (stride * 8) == 0 and width % (stride * 8) == 0, 'height and width must be divisible by the stride multiplied by 8'
        assert stride % 8 == 0 and stride < 64, 'stride must be divisible by 8 and smaller than the latent size of Stable Diffusion'

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # obtain text embeddings
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # define a list of windows to process in parallel
        views = get_views(height, width, stride=stride)
        print(f"[INFO] number of views to process: {len(views)}")
        
        # Initialize latent
        latent = torch.randn((1, self.unet.in_channels, height // 8, width // 8))

        count = torch.zeros_like(latent, requires_grad=False, device=self.device)
        value = torch.zeros_like(latent, requires_grad=False, device=self.device)
        latent = latent.to(self.device)
        
        # set DDIM scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        
        # set the anchor view as the middle view
        anchor_view_idx = len(views) // 2

        # set SyncDiffusion scheduler
        sync_scheduler = exponential_decay_list(
            init_weight=sync_weight,
            decay_rate=sync_decay_rate,
            num_steps=num_inference_steps
        )
        print(f'[INFO] using exponential decay scheduler with decay rate {sync_decay_rate}')

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                count.zero_()
                value.zero_()

                '''
                (1) First, obtain the reference anchor view (for computing the perceptual loss)
                '''
                with torch.no_grad():
                    if (i + 1) % sync_freq == 0 and i < sync_thres:
                        # decode the anchor view
                        h_start, h_end, w_start, w_end = views[anchor_view_idx]
                        latent_view = latent[:, :, h_start:h_end, w_start:w_end].detach()

                        latent_model_input = torch.cat([latent_view] * 2)                                               # 2 x 4 x 64 x 64
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                        # perform guidance
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred_new = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        # predict the 'foreseen denoised' latent (x0) of the anchor view
                        latent_pred_x0 = self.scheduler.step(noise_pred_new, t, latent_view)["pred_original_sample"]
                        decoded_image_anchor = self.decode_latents(latent_pred_x0)                                      # 1 x 3 x 512 x 512

                '''
                (2) Then perform SyncDiffusion and run a single denoising step
                '''
                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end].detach()

                    ############################## BEGIN: PERFORM GRADIENT DESCENT (SyncDiffusion) ##############################
                    latent_view_copy = latent_view.clone().detach()

                    #### TODO: TEST ####
                    # if i % sync_freq == 0 and i < sync_thres:
                    if (i + 1) % sync_freq == 0 and i < sync_thres:
                        
                        # gradient on latent_view
                        latent_view = latent_view.requires_grad_()

                        # expand the latents for classifier-free guidance
                        latent_model_input = torch.cat([latent_view] * 2)

                        # predict the noise residual
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                        # perform guidance
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred_new = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        # compute the denoising step with the reference model
                        out = self.scheduler.step(noise_pred_new, t, latent_view)

                        # predict the 'foreseen denoised' latent (x0)
                        latent_view_x0 = out['pred_original_sample']

                        # decode the denoised latent
                        decoded_x0 = self.decode_latents(latent_view_x0)                 # 1 x 3 x 512 x 512
                        
                        # compute the perceptual loss (LPIPS)
                        percept_loss = self.percept_loss(
                            decoded_x0 * 2.0 - 1.0, 
                            decoded_image_anchor * 2.0 - 1.0
                        )

                        # compute the gradient of the perceptual loss w.r.t. the latent
                        norm_grad = grad(outputs=percept_loss, inputs=latent_view)[0]

                        # SyncDiffusion: update the original latent
                        if view_idx != anchor_view_idx:
                            latent_view_copy = latent_view_copy - sync_scheduler[i] * norm_grad                             # 1 x 4 x 64 x 64   
                    ############################## END: PERFORM GRADIENT DESCENT (SyncDiffusion) ##############################
                    
                    # after gradient descent, perform a single denoising step
                    with torch.no_grad():
                        latent_model_input = torch.cat([latent_view_copy] * 2)
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred_new = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        out = self.scheduler.step(noise_pred_new, t, latent_view_copy)
                        latent_view_denoised = out['prev_sample'] 

                    # merge the latent views
                    value[:, :, h_start:h_end, w_start:w_end] += latent_view_denoised
                    count[:, :, h_start:h_end, w_start:w_end] += 1

                # take the MultiDiffusion step (average the latents)
                latent = torch.where(count > 0, value / count, value)

        # decode latents to panorama image
        with torch.no_grad():
            imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
            img = T.ToPILImage()(imgs[0].cpu())

        print(f"[INFO] Done!")

        return img
