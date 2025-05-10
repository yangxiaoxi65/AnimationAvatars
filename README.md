# Animation Avatars

A pipeline for generating animatable avatars using in-the-wild video sequence for Brown CSCI 1430 Computer Vision course

Tianhao Shi, Xiaoxi Yang, Hayden McDonald, Nicholas Marsano

## Requirements

- 64-bit Linux
- NVIDIA GPU with CUDA 12.x 
- Conda support

## Installation

1. Clone the repository

```bash
git clone --recursive https://github.com/yangxiaoxi65/AnimationAvatars.git
```

2. Setup environment

```bash
cd AnimationAvatars
conda env create -f environment.yml
conda activate animation_avatars
```
3. Install OpenPose and build it from source


Due to the extreme finickiness of OpenPose, we refer to the [original repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose.git) for installation instructions.

After installing, make sure to add the path to `openpose.bin` to your `PATH` environment variable. For example,

```bash
which openpose.bin # /path/to/openpose/build/examples/openpose/openpose.bin
export PATH=$PATH:/path/to/openpose/build/examples/openpose
```

4. 