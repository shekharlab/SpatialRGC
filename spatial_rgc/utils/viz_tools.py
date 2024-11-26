import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import importlib
import scanpy as sc
import re 
import pandas as pd

from matplotlib.lines import Line2D
import matplotlib as mpl

import spatial_rgc.utils.rconstants as rc
import spatial_rgc.utils.aligner as aligner


def create_aligner(alignment,stain,im_dir):
    f_paths = []
    centers= {} 
    rotations=[]
    for i,subdirectory in enumerate(alignment):
        if "center" in alignment[subdirectory]:
            center = alignment[subdirectory]["center"]
        else:
            centers=None
        centers[i] = center
        dr =  alignment[subdirectory]["downsample_ratio"]
        z = alignment[subdirectory]["z"]
        f = os.path.join(im_dir, subdirectory, f"mosaic_{stain}_z{z}_dr{dr}.tif")
        f_paths.append(f)
        rotations.append(alignment[subdirectory]["rotation"]) # Rotations is in terms of micron pic
    rotations = np.array(rotations)
    

    a=aligner.Aligner(f_paths,centers=centers,rotations=360-rotations) # Convert to img_rotation
    return a

def _extract_dir_region(subdirectory,base_dir): #base_dir is abs path to dir containing subdirectory as child
    num_genes,run,region = subdirectory.split("_")
    run_dir= num_genes + "_" + run
    figure_dir = os.path.join(base_dir,run_dir)
    if not os.path.isdir(figure_dir):
        os.mkdir(figure_dir)

    return figure_dir

def visualize_image(alignment, stain, indices,im_dir,visualize=True,overlay=False,figure_dir="spatial_figures",run="140g_rn3"):
    a = create_aligner(alignment,stain,im_dir)
    if visualize:
        a.visualize(indices)
    if overlay:
        output_dir = _extract_dir_region(a.subdirectories[0],figure_dir)
        save_f = os.path.join(output_dir,f"aligned_{stain}.png")
        a.overlay_images(0,indices,save_f,run=run)

def rotation_matrix(rotat_degree):
    rad =  rotat_degree/180*np.pi
    return np.array([[np.cos(rad), -np.sin(rad)],[np.sin(rad),np.cos(rad)]])

