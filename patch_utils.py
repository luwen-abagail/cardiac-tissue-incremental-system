import math
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
import torch.nn.functional as F
import math
import itertools
import random

import torch.nn as nn


import numpy as np
import torch
import itertools

from look import newnii,newii

def _get_patches(imgs, masks, classes=4, img_size=256, patch_size=64, num_patches_per_class=10):
    """
    INPUTS:
    'classes' (int): the number of classes in the dataset
    'imgs' (tensor): input images of shape (batch X channels X img_size X img_size)
    'masks' (tensor): input masks of shape (batch X 'classes' X 'img_size' X 'img_size')
    'num_patches_per_class' (int): number of patches to sample per class

    OUTPUTS:
    'centers' (list of lists): list of len 'classes' made of lists of center coordinates (y, x) 
             for each patch, shape (num_patches_per_class X 2)
    """
    centers = []
    
    batch_size, _, _, _ = imgs.shape
    
    for b in range(batch_size):
        # Convert one-hot to class indices
        mask = masks[b].cpu().numpy()
        class_centers = {c: [] for c in range(classes)}
        
        # Extract all candidate points for each class
        for c in range(classes):
            candidate_points = list(zip(*np.where(mask == c)))
            for y, x in candidate_points:
                # Check if the patch is within image bounds
                if (0 <= y - patch_size//2 and y + patch_size//2 <= img_size and 
                    0 <= x - patch_size//2 and x + patch_size//2 <= img_size):
                    # Calculate patch center (adjusting for patch size)
                    center_y = y
                    center_x = x
                    class_centers[c].append((center_y, center_x))
        
        # Resample to ensure each class has exactly 'num_patches_per_class' centers
        resampled_centers = [[] for _ in range(classes)]
        for c in range(classes):
            if len(class_centers[c]) > num_patches_per_class:
                # Randomly sample centers if there are more than needed
                resampled_centers[c] = random.sample(class_centers[c], num_patches_per_class)
            else:
                if len(class_centers[c]) > 0:
                    # If not enough centers, randomly sample with replacement
                    resampled_centers[c] = random.choices(class_centers[c], k=num_patches_per_class)
                else:
                    # If no centers available, return empty list
                    resampled_centers[c] = []
        
        return resampled_centers

def get_embeddings(feature, patch_list, patch_size,patchnum):
    # We'll combine features from all layers
    all_embeddings = []
    #print("len(patch_list)",len(patch_list),patch_size,patchnum)
    for numclass in range(len(patch_list)):
        layer_embeddings = []
        for point in range(patchnum):
            
            cls_layer_emb = []
           
            for cls in range(len(feature)):
                layer_feat=feature[cls]
                if len(patch_list[numclass])==0:
                    continue
                if patch_list[numclass][point]:
                    y, x = patch_list[numclass][point]
                    #print(y,x,"y,x")
                    # Calculate patch boundaries
                    y_min = y - patch_size // 2
                    y_max = y + patch_size // 2
                    x_min = x - patch_size // 2
                    x_max = x + patch_size // 2
                    
                    # Extract patch
                    patch = layer_feat[:, :, y_min//(2**(cls)):y_max//(2**(cls)), x_min//(2**(cls)):x_max//(2**(cls))]
                    #print(patch.shape)
                    cls_layer_emb.append(patch)
            layer_embeddings.append(cls_layer_emb)
        all_embeddings.append(layer_embeddings)
        #print("all_embeddings",len(all_embeddings),len(all_embeddings[0]),len(all_embeddings[0][0]))
    return all_embeddings