import os
from os.path import join
import numpy as np
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from torchmetrics.multimodal import CLIPScore

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

seed_everything(2023)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(data_dir, prompt, save_path):
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    metric = metric.to(device)

    image_paths = os.listdir(data_dir)

    clip_score_list = []

    print(f"[INFO] Number of images: {len(image_paths)}")
    print(f"[INFO] Prompt: {prompt}")

    for image_path in image_paths:
        image = Image.open(join(data_dir, image_path))
        image = T.ToTensor()(image)
        image = image.to(device)
        image = image * 255.0
        image = image.type(torch.int64)
        
        # Compute CLIP score
        score = metric(image, prompt)
        clip_score_list.append(score.item())

    print("# --------------------------------- #")
    print(f"[INFO] CLIP score (avg): {np.mean(clip_score_list)}")
    print(f"[INFO] CLIP score (avg): {np.std(clip_score_list)}")
    print("# --------------------------------- #")

    with open(save_path, "a") as f:
        f.write(f"avg: {round(np.mean(clip_score_list), 4)}\n")
        f.write(f"std: {round(np.std(clip_score_list), 4)}\n")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./image_dir',
                        help='Directory containing images.')
    parser.add_argument('--save_path', type=str, default='clip_score.txt',
                        help='Path to save the CLIP score.')
    parser.add_argument('--prompt', type=str, default='a photo of a city skyline at night',
                        help='Prompt to use with CLIP.')
    args = parser.parse_args()

    main(args.data_dir, args.prompt, args.save_path)