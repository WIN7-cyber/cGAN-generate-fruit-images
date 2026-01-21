# UV IMDA Project

## Image Generation from Labels using ACGAN

---

## 1. Dataset

### Original Dataset
Fruits Dataset (Kaggle):  
https://www.kaggle.com/datasets/moltean/fruits  

Each class in the original dataset contained only about **500 images**.  
To address this limitation, we merged classes corresponding to the **same fruit**.

After preprocessing, we selected **11 fruit categories**, with approximately **1,100 images per category**.

### Preprocessed Dataset
The final preprocessed dataset is available at:  
https://drive.google.com/drive/folders/1J037ySeZ0z3Ljbr_USWy1OaDhRrRUrtB?usp=sharing  

---

## 2. Model

- **Model:** ACGAN (Auxiliary Classifier GAN)
- **Version:** Basic / baseline implementation  

This version is used to visualize and evaluate the best generated results.

---

## 3. Environment & Imports

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils

## 4. Data Preprocessing & Image Size

Two different preprocessing pipelines were used in two notebooks.

### ACGAN.ipynb

- Images are resized directly to **128 × 128**
- Transformations:
  - `RandomHorizontalFlip`
  - `ToTensor`
  - `Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])`

---

### fruit.ipynb

- Images are resized to **72 × 72**, then cropped using `RandomCrop(64)`
- This introduces more **data augmentation**
- Final image resolution: **64 × 64**
- Transformations:
  - `RandomHorizontalFlip`
  - `ToTensor`
  - `Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])`

---

## 5. Comparison of Preprocessing Strategies

- **ACGAN.ipynb**
  - Simple resizing
  - Higher resolution (**128 × 128**)

- **fruit.ipynb**
  - Resize + RandomCrop for increased data diversity
  - Lower final resolution (**64 × 64**)
  - Stronger data augmentation

---

