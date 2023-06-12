# SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions

![demo](./docs/demo/sync_demo.gif)

[**Arxiv**](https://arxiv.org/abs/2306.05178) | [**Project Page**](https://syncdiffusion.github.io/) <br>

[Yuseung Lee](https://phillipinseoul.github.io/), [Kunho Kim](), [Hyunjin Kim](), [Minhyuk Sung](https://mhsung.github.io/) <br>

**We plan to release our code soon. Please stay tuned! :hugs:**

# Introduction
This repository contains the official implementation of **SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions**.<br>
**SyncDiffusion** is a plug-and-play module that synchronizes multiple diffusions through gradient descent from a perceptual similarity loss. More results can be viewed on our [project page](https://syncdiffusion.github.io/).

[//]: # (### Abstract)
> The remarkable capabilities of pretrained image diffusion models have been utilized not only for generating fixed-size images but also for creating panoramas. However, naive stitching of multiple images often results in visible seams. Recent techniques have attempted to address this issue by performing joint diffusions in multiple windows and averaging latent features in overlapping regions. However, these approaches, which focus on seamless montage generation, often yield incoherent outputs by blending different scenes within a single image. To overcome this limitation, we propose SyncDiffusion, a plug-and-play module that synchronizes multiple diffusions through gradient descent from a perceptual similarity loss. Specifically, we compute the gradient of the perceptual loss using the predicted denoised images at each denoising step, providing meaningful guidance for achieving coherent montages. Our experimental results demonstrate that our method produces significantly more coherent outputs compared to previous methods (66.35% vs. 33.65% in our user study) while still maintaining fidelity (as assessed by GIQA) and compatibility with the input prompt (as measured by CLIP score).

# Citation
If you find our work useful, please consider citing:
```
@article{lee2023syncdiffusion,
    title={SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions}, 
    author={Yuseung Lee and Kunho Kim and Hyunjin Kim and Minhyuk Sung},
    journal={arXiv preprint arXiv:2306.05178},
    year={2023}
}
```

# Acknowledgement
Our code is heavily based on the [official implementation](https://github.com/omerbt/MultiDiffusion) of [MultiDiffusion](https://multidiffusion.github.io/). We borrowed the Github template from [SALAD](https://github.com/KAIST-Geometric-AI-Group/SALAD).
