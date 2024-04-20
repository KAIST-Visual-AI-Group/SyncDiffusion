import os
import lpips
import torch
import cv2
import numpy as np
import itertools
import argparse

def get_views_crop(panorama_height, panorama_width, window_size=512, stride=512):
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  

# ----------------------------------------------------------- #

seed_everything(2023)
device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    image_paths = os.listdir(args.data_dir)
    
    # Load LPIPS
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    loss_fn_vgg.to(device)

    intra_lpips_list = []

    for image_idx, image_path in enumerate(image_paths):
        # Load panorama image
        panorama = cv2.imread(os.path.join(args.data_dir, image_path))       # Panorama image
        panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
        panorama = panorama.astype(np.float32) / 255.0
        panorama = panorama * 2.0 - 1.0                                 # normalize to [-1, 1]
        H, W, _ = panorama.shape

        # Obtain crop positions
        crop_pos_list = get_views_crop(H, W, window_size=512, stride=args.stride)
        crops_torch = torch.zeros((len(crop_pos_list), 3, 512, 512)) # .to(device)

        # Obtain crops in torch
        for view_idx, (h_start, h_end, w_start, w_end) in enumerate(crop_pos_list):
            crop = panorama[h_start:h_end, w_start:w_end]
            crops_torch[view_idx] = torch.from_numpy(crop.transpose(2, 0, 1))
        crops_torch = crops_torch.to(device)

        idx_list = range(len(crop_pos_list))
        combs = list(itertools.combinations(idx_list, 2))

        # compute LPIPS
        intra_lpips = 0.0

        for idx, comb in enumerate(combs):
            crop1 = crops_torch[comb[0]]                                # 3 x 512 x 512
            crop2 = crops_torch[comb[1]]                                # 3 x 512 x 512
            intra_lpips += loss_fn_vgg(crop1, crop2).item()
        
        # Compute average
        intra_lpips /= len(combs)
        intra_lpips_list.append(intra_lpips)

        if (image_idx + 1) % 100 == 0:
            avg_intra_lpips = round(np.mean(intra_lpips_list[image_idx + 1 - 100 : image_idx + 1]), 4)
            print(f"[INFO] Avg. Intra LPIPS ({image_idx + 1 - 100}-{image_idx}): {avg_intra_lpips}")

            with open(args.save_path, "a") as f:
                f.write(f"Avg. Intra LPIPS ({image_idx + 1 - 100}-{image_idx}): {avg_intra_lpips}\n")

    # Compute the average
    intra_lpips_full = np.mean(intra_lpips_list)
    print(f"[INFO] Avg. Intra LPIPS: {round(intra_lpips_full, 4)}")

    with open(args.save_path, "a") as f:
        f.write(f"[INFO] Avg. Intra LPIPS of {args.data_dir}: {round(intra_lpips_full, 4)}\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./pano_dir", help="Directory containing panorama images")
    parser.add_argument("--save_path", type=str, default="intra_lpips.txt", help="Path to save the Intra-LPIPS score")
    parser.add_argument("--stride", type=int, default=512, help="Cropping stride for the panorama image")
    args = parser.parse_args()

    main(args)