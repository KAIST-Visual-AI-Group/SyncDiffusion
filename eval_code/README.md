## Evaluation Code
First, randomly crop the generated panorama images into `512x512` images for evaluation. (NOTE: Tested on `512x3017 (1:6)` panorama images.)
```
$ crop_panorama.py \
--num_crop 1 \            # Num. of crops from per panorama
--data_dir $PANO_DIR \    # Directory of generated panoramas
--save_dir $IMAGE_DIR     # Directory for cropped images
```

### LPIPS (Intra-LPIPS)
Compute the Intra LPIPS with the original panorama images in `$PANO_DIR`.
```
$ python eval_intra_lpips.py \
--data_dir $IMAGE_DIR \     # Directory of panorama images
--save_path ./intra_lpips.txt \
--stride 512
```

### CLIP Score (Mean-CLIP-S)
Install `torchmetrics` by
```
$ pip install torchmetrics
```
Compute the CLIP score with the cropped images in `$IMAGE_DIR`.
```
$ python eval_clip_score.py \
--data_dir $IMAGE_DIR \     # Directory of cropped images
--save_path ./clip_score.txt \
--prompt $TEXT_PROMPT
```

### Style Loss (Intra-Style-L)
Intra-Style-L is measured based on [this](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#neural-transfer-using-pytorch) codebase originally from [Gatys et al.](https://arxiv.org/abs/1508.06576). 
```
$ eval_style_loss.py \
--data_dir $PANO_DIR \          # Directory of panorama images
--save_path ./intra_style_loss.txt \
--stride 512
```

### FID, KID
FID, KID are measured with the `clean-fid` library. For details, visit https://github.com/GaParmar/clean-fid.
```
pip install clean-fid
```
The refence images are generated with the same version of [Stable Diffusion v2.0-base](https://huggingface.co/stabilityai/stable-diffusion-2-base).

### GIQA (Mean-GIQA)
Mean-GIQA is measured based on the official [GIQA](https://github.com/cientgu/GIQA) code. We use the `KNN-GIQA` with `K=8`. The GIQA score of each cropped image in `$IMAGE_DIR` is computed and is averaged to obtain the Mean-GIQA of the generated images.