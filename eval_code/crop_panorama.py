'''
Crops panorama images to 512 x 512 images
for evaluation.
'''
import os
from os.path import join
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import torch

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  

seed_everything(2023)
STRIDE = 16     # Stride for cropping

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # Get all images in the data directory
    images = [
        f for f in os.listdir(args.data_dir) 
        if ('.png' in f or '.jpg' in f)
    ]

    # Crop panorama images to 512 x 512
    for image_name in tqdm(images):
        image = cv2.imread(join(args.data_dir, image_name))
        H, W, _ = image.shape
        
        assert H == 512, f"Panorama height is not 512: {H}"

        for _ in range(args.num_crops):
            rand_int = np.random.randint(0, W - 512)
            image_cropped = image[:, rand_int:rand_int + 512, :]    # Random crop

            cv2.imwrite(join(args.save_dir, image_name), image_cropped)
    
    print(f"[INFO] Cropped images saved to {args.save_dir}.")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.', help='path to panorama folder')
    parser.add_argument('--save_dir', type=str, default='.', help='path to output folder')
    parser.add_argument('--num_crops', type=int, default=1, help='number of crops per image')
    args = parser.parse_args()

    main(args)