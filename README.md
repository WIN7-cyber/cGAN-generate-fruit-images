Project

Image Generation from Labels (ACGAN)
1. Dataset

Original Dataset:
Fruits Dataset from Kaggle
https://www.kaggle.com/datasets/moltean/fruits

Since each class in the original dataset contained only around 500 images, we merged classes corresponding to the same fruit.

After preprocessing, we selected 11 fruit categories, each containing approximately 1,100 images.

Preprocessed Dataset (Final Version):
https://drive.google.com/drive/folders/1J037ySeZ0z3Ljbr_USWy1OaDhRrRUrtB?usp=sharing

2. Model

Model used: ACGAN (Auxiliary Classifier GAN)

Version: Basic / baseline implementation

This version is finally used to visualize and evaluate the best results.

3. Environment & Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils

4. Data Preprocessing & Image Size

We experimented with two different preprocessing pipelines in two notebooks:

ACGAN.ipynb

Resize images directly to 128 × 128

Transformations:

RandomHorizontalFlip

ToTensor

Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

fruit.ipynb

Resize images to 72 × 72, then apply RandomCrop(64)

This introduces more data augmentation

Final image resolution: 64 × 64

Transformations:

RandomHorizontalFlip

ToTensor

Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

5. Key Differences Between the Two Approaches

ACGAN.ipynb

Simple resizing

Higher resolution (128 × 128)

fruit.ipynb

Resize + RandomCrop for greater data diversity

Lower final resolution (64 × 64)

Stronger data augmentation
