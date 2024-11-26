import pandas as pd
import numpy as np
import scanpy as sc
from cellpose import utils
from cellpose.io import imread
import cv2

import multiprocessing
from multiprocessing import Pool
import argparse
import os
from timeit import default_timer as timer

from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--trial_name", help="Name of segmentation file",type=str)
parser.add_argument("-c", "--channels",nargs='+',help="Stain channels to create new columns in anndata object", type=str,default=None)
parser.add_argument("-z", "--zlayers",nargs='+', type=int,default=None)
parser.add_argument("-s", "--downsample_ratio", help="downsample factor to use", type=int,default=1)
parser.add_argument("-S", "--subdirectory", type=str, default=None)
parser.add_argument("-m", "--max_z", action='store_true', default=False)


args = parser.parse_args()
OUTPUT_DIR = os.path.join(BASE_DIR,"outputs", args.subdirectory)
IM_DIR= os.path.join(BASE_DIR,"images", args.subdirectory)
SEG_F = os.path.join(OUTPUT_DIR, f"stitched_masks_{args.trial_name}_comb.npy")
SEG_F_MAXZ= os.path.join(OUTPUT_DIR, f"{args.trial_name}_mosaic_maxz.npy")


# updates c_intensity and c_counts in place; expects 2d matrix
def _update_stain(mask_mat,img,intensities,n_counts):
    for y in range(mask_mat.shape[0]):
        for x in range(mask_mat.shape[1]):
            id=mask_mat[y,x]
            intensities[str(id)] += img[y,x] # needs to be string to map to indices in adata 
            n_counts[str(id)] +=1
        

def add_costain(trial_name,channels,zlayers, downsample_ratio,seg_f, output_dir,im_dir):
    f=os.path.join(output_dir, f"Cellpose_model_{trial_name}.h5ad")
    adata = sc.read_h5ad(f)
    channel_intensities = dict( zip(channels, [defaultdict(lambda: 0)] * len(channels) ) )
    channel_counts = dict(zip(channels,[defaultdict(lambda: 0)] * len(channels) ))
    mask_mat = np.load(seg_f,allow_pickle=True).astype(int)

    for c in channels:
        c_intensity = channel_intensities[c]
        c_counts = channel_counts[c]
        
        for z in zlayers:
            im_f= os.path.join(im_dir,f"mosaic_{c}_z{z}.tif")
            assert os.path.exists(im_f), f"Data file {im_f} is not present for given channel: {c}' and z_stack: {z}"
            img = imread(im_f)
    
            if args.downsample_ratio != 1:
                img = cv2.resize(img,(img.shape[1]//downsample_ratio, img.shape[0]//downsample_ratio))
                
            z_l = mask_mat[z]
            _update_stain(mask_mat=z_l,img=img,intensities=c_intensity,n_counts=c_counts)
            print(f"Finished checking intensity in z{z} for stain {c}",flush=True)
        
        for sid in c_intensity: # Normalize by total number of pixels
            c_intensity[sid] /= c_counts[sid]
        
        adata.obs[c] = pd.Series(c_intensity)
        print(f"Note: background intensity for channel {c} is {c_intensity['0']}",flush=True)

    adata.write_h5ad(os.path.join(output_dir,f"Cellpose_model_{trial_name}.h5ad"))

def add_costain_maxz(trial_name,channels,zlayers, downsample_ratio,seg_f, output_dir,im_dir):
    f=os.path.join(output_dir, f"Cellpose_model_maxz_{trial_name}.h5ad")
    adata = sc.read_h5ad(f)
    channel_intensities = dict( zip(channels, [defaultdict(lambda: 0)] * len(channels) ) )
    channel_counts = dict(zip(channels,[defaultdict(lambda: 0)] * len(channels) ))
    mask_mat = np.load(seg_f,allow_pickle=True).astype(int)
    assert len(mask_mat.shape) == 2

    for c in channels:
        c_intensity = channel_intensities[c]
        c_counts = channel_counts[c]
        max_img=None
        for z in args.zlayers:
            fname = os.path.join(im_dir, f"mosaic_{c}_z{z}.tif")
            assert os.path.exists(fname), f"Data file {fname} is not present for given channel: '{c}' and z_stack: {z}"
            img = imread(fname)
    
            if max_img is None:
                max_img = img
            else:
                max_img = np.maximum(max_img,img)
                
        if args.downsample_ratio != 1:
            max_img = cv2.resize(max_img,(max_img.shape[1]//downsample_ratio, max_img.shape[0]//downsample_ratio))
                
        _update_stain(mask_mat=mask_mat,img=max_img,intensities=c_intensity,n_counts=c_counts)
        print(f"Finished checking intensity in maxz for stain {c}",flush=True)
        
        for sid in c_intensity: # Normalize by total number of pixels
            c_intensity[sid] /= c_counts[sid]
        
        adata.obs[c] = pd.Series(c_intensity)
        print(f"Note: background intensity for channel {c} is {c_intensity['0']}",flush=True)

    adata.write_h5ad(os.path.join(output_dir,f"Cellpose_model_maxz_{trial_name}.h5ad"))
        
        
        
        

if __name__=="__main__":
    start = timer()
    if args.channels is not None:
        if args.max_z:
            add_costain_maxz(trial_name=args.trial_name, channels=args.channels,zlayers=args.zlayers, downsample_ratio=args.downsample_ratio,seg_f=SEG_F_MAXZ, output_dir=OUTPUT_DIR,im_dir=IM_DIR)
        else:
            add_costain(trial_name=args.trial_name, channels=args.channels,zlayers=args.zlayers, downsample_ratio=args.downsample_ratio,seg_f=SEG_F, output_dir=OUTPUT_DIR,im_dir=IM_DIR)
        end=timer()
    else:
        print("No channels, not doing anything")
    print("Time elapsed: ", end-start)