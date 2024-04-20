'''
Original code from:
- https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#neural-transfer-using-pytorch
Based on:
- https://arxiv.org/abs/1508.06576
'''
import os
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import torch.optim as optim
import itertools
import argparse
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  

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

# ------------------------ Style Loss ------------------------ #
def gram_matrix(input):
    a, b, c, d = input.size()  
    # a: batch size (=1)
    # b: number of feature maps
    # (c,d): dimensions of a f. map (N = c*d)

    # Resize F_XL into \hat F_XL
    features = input.view(a * b, c * d) 

    # Compute the gram product
    G = torch.mm(features, features.t()) 

    # We 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std
        
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer

def get_style_model(
        cnn, 
        normalization_mean, 
        normalization_std,
        style_img,
        style_layers
    ):
    # Normalization module
    normalization = Normalization(
        normalization_mean, 
        normalization_std
    ).to(device)

    # losses
    style_loss_list = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers:
            # Add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_loss_list.append(style_loss)

    # Now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_loss_list
# ------------------------ Style Loss ------------------------ #

loader = T.Compose([
        T.Resize(IMAGE_SIZE),  # Scale the image so that the shorter side is 512
        T.ToTensor()
    ])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Measure the Intra Style Loss for the panorama images
def main(
    data_dir, 
    save_path, 
    stride
):
    image_paths = os.listdir(data_dir)
    
    # Load VGG19 model
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    style_loss_list = []

    # Iterate for all images
    for image_idx, image_path in enumerate(tqdm(image_paths)):
        pano_image = image_loader(join(data_dir, image_path))   # (1, 3, 512, 512)

        H, W = pano_image.shape[2], pano_image.shape[3]

        # Obtain crop positions
        crop_pos_list = get_views_crop(H, W, window_size=512, stride=stride)

        # Get the cropped images
        crops_torch = torch.zeros((len(crop_pos_list), 1, 3, 512, 512)) # .to(device)

        for view_idx, (h_start, h_end, w_start, w_end) in enumerate(crop_pos_list):
            crops_torch[view_idx] = pano_image[:, :, h_start:h_end, w_start:w_end]
        crops_torch = crops_torch.to(device)

        idx_list = range(len(crop_pos_list))
        combs = list(itertools.combinations(idx_list, 2))

        style_loss = 0.0

        for idx, comb in enumerate(combs):
            crop1 = crops_torch[comb[0]]                                # 3 x 512 x 512
            crop2 = crops_torch[comb[1]]                                # 3 x 512 x 512

            style_model, style_losses = get_style_model(
                cnn, 
                cnn_normalization_mean, 
                cnn_normalization_std,
                crop1,
                style_layers=style_layers_default
            )

            # compute style loss
            crop2.requires_grad_(True)
            style_model.requires_grad_(False)

            optimizer = get_input_optimizer(crop2)

            with torch.no_grad():
                crop2.clamp_(0, 1)

            optimizer.zero_grad()
            style_model(crop2)

            style_score = 0

            for sl in style_losses:
                style_score += sl.loss
            style_loss += style_score.item()

            style_model = None
            torch.cuda.empty_cache()

        style_loss_list.append(style_loss)
    
    # Average style loss
    style_loss_avg = np.mean(style_loss_list)
    print(f"[INFO] Intra Style Loss (avg): {style_loss_avg}")

    with open(save_path, 'a') as f:
        f.write(f"Intra Style Loss (avg): {style_loss_avg}\n")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./pano_dir", help="Directory containing panorama images")
    parser.add_argument("--save_path", type=str, default="intra_style_loss.txt", help="Path to save the Intra Style Loss score")
    parser.add_argument("--stride", type=int, default=512, help="Cropping stride for the panorama image")
    args = parser.parse_args()

    seed_everything(2023)

    main(args.data_dir, args.save_path, args.stride)