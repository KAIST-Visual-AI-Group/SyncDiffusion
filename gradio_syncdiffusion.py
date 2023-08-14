import os
from os.path import join
import gradio as gr
from datetime import datetime
import time
import torch

from syncdiffusion.syncdiffusion_model import SyncDiffusion
from syncdiffusion.utils import seed_everything

device = "cuda" if torch.cuda.is_available() else "cpu"  

# Load SyncDiffusion model
syncdiffusion = SyncDiffusion(device, sd_version="2.0")

def process(
    prompts, 
    height, 
    width, 
    sync_weight,
    sync_decay_rate,
    sync_freq,
    sync_thres,
    seed
):
    seed_everything(seed)

    img = syncdiffusion.sample_syncdiffusion(
        prompts = prompts,
        negative_prompts = "",
        height = height,
        width = width,
        num_inference_steps = 50,
        guidance_scale = 7.5,
        sync_weight = sync_weight,
        sync_decay_rate = sync_decay_rate,
        sync_freq = sync_freq,
        sync_thres = sync_thres,
        stride = 16
        )
    return [img]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("SyncDiffusion Text-to-Panorama Demo")

    with gr.Row():
        with gr.Column():
            run_button = gr.Button(label="Run")

            prompt = gr.Textbox(label="Text Prompt", value='a cinematic view of a castle in the sunset')
            sync_weight = gr.Slider(label="Sync Weight", minimum=0.0, maximum=30.0, value=20.0, step=5.0)
            sync_decay_rate = gr.Slider(label="Sync Decay Rate", minimum=0.0, maximum=1.0, value=0.99, step=0.01)
            sync_freq = gr.Slider(label="Sync Frequency", minimum=1, maximum=50, value=1, step=1)
            sync_thres = gr.Slider(label="Sync Threshold", minimum=0, maximum=50, value=10, step=0.01)

            width = gr.Slider(label="Width", minimum=512, maximum=4096, value=2048, step=128)
            height = gr.Slider(label="Height", minimum=512, maximum=4096, value=512, step=128)

            seed = gr.Number(label="Seed", value=0)

        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')

    ips = [prompt, height, width, sync_weight, sync_decay_rate, sync_freq, sync_thres, seed]

    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')

