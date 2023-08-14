import os
from os.path import join
from datetime import datetime
import time
import torch
import argparse

import sys
from syncdiffusion.syncdiffusion_model import SyncDiffusion
from syncdiffusion.utils import seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='natural landscape in anime style illustration')
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=3072)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--sync_weight', type=float, default=0.1, help="weight for SyncDiffusion")
    parser.add_argument('--sync_thres', type=int, default=40, help="max step for SyncDiffusion")
    parser.add_argument('--sync_freq', type=int, default=1, help="frequency for SyncDiffusion")
    parser.add_argument('--stride', type=int, default=8, help="window stride for MultiDiffusion")
    parser.add_argument('--sync_decay_rate', type=float, default=0.99, help="SyncDiffusion weight scehduler decay rate")
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    prompt_full = "_".join(args.prompt.replace(",", " ").split(" "))
    save_dir_full = join(args.save_dir, prompt_full)
    os.makedirs(save_dir_full, exist_ok=True)

    # Load SyncDiffusion model
    syncdiffusion_model = SyncDiffusion(device, sd_version=args.sd_version)

    seed_everything(args.seed)

    # Generate images
    img = syncdiffusion_model.sample_syncdiffusion(
        prompts = args.prompt,
        negative_prompts = args.negative,
        height = args.H,
        width = args.W,
        num_inference_steps = args.steps,
        guidance_scale = 7.5,
        sync_weight = args.sync_weight,
        sync_decay_rate = args.sync_decay_rate,
        sync_freq = args.sync_freq,
        sync_thres = args.sync_thres,
        stride = args.stride
    )
    img.save(join(save_dir_full, f"sample_seed_{args.seed:06d}.png"))
    print(f"[INFO] saved the result for prompt: {args.prompt}")


if __name__ == "__main__":
    main()