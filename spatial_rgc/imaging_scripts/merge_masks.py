import scanpy as sc
import pandas as pd
import argparse
import datetime
import gc
import logging
from statistics import multimode
import numpy as np
import os
import time
import multiprocessing
from multiprocessing import Process, allow_connection_pickling

from cellpose import io
from cellpose import models
from cellpose.io import imread
import cv2
import gc

VALID_CHANNELS = ["DAPI", "Cellbound2"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLD = "models"
MODELS_DIR = os.path.join(BASE_DIR,MODELS_FOLD)

parser = argparse.ArgumentParser()
parser.add_argument("-z", "--zlayers", help="Z_layers to use", nargs='+', type=int, default=[0,1,2,3,4,5,6])

parser.add_argument("-x", "--max_z", help='Use max_z instead of stitching',action='store_true',default=False)

parser.add_argument("-c", "--channel", help="channel", type=str)
parser.add_argument("-n", "--trial_name", help="channel", type=str)
parser.add_argument("-s", "--downsample_ratio", help="channel", type=int,default=1)
parser.add_argument("-d", "--diameter", help="Diameter size for model",type=float, default=100)
parser.add_argument("-f", "--flow_threshold",type=float, default=0.4)
parser.add_argument("-P", "--cellprob_threshold", type=float, default=0.0)
parser.add_argument("-p", "--num_processes", type=int, default=1)
parser.add_argument("-b", "--boundaries", nargs='+',type=int, default=None) #Note:[5 10 15 20] cooresponds to img[5:10][15:20]
parser.add_argument("-r", "--resegment", action='store_true',default=False)
parser.add_argument("-m", "--merge", action='store_true',default=False)
parser.add_argument("-g", "--use_gpu", action='store_true',default=False)
parser.add_argument("-M", "--model_name", type=str,default=None)   
parser.add_argument("-S", "--subdirectory", type=str,default="")
parser.add_argument("-a", "--augment_with_transcripts",help="add transcripts on top of image", action='store_true',default=False)


args = parser.parse_args()
if args.model_name is None:
    model = models.Cellpose(model_type='cyto2',gpu=args.use_gpu)
    print("Using model: cyto2",flush=True)
else:
    model = models.CellposeModel(pretrained_model=os.path.join(MODELS_DIR, args.model_name),gpu=args.use_gpu)
    print(f"Using model: {args.model_name}",flush=True)

IMAGES_DIR = os.path.join(BASE_DIR,"images", args.subdirectory)
FIGURE_DIR = os.path.join(BASE_DIR,"figures", args.subdirectory)
OUTPUT_DIR = os.path.join(BASE_DIR,"outputs", args.subdirectory)
MAPPING_DIR = os.path.join(BASE_DIR,"mappings",args.subdirectory)


def _augment(img,transcripts_f="detected_transcripts.csv",downsample_ratio=4,transform_f="micron_to_mosaic_pixel_transform.csv"):
    print("Augmenting maxz image with transcripts",flush=True)
    tdf = pd.read_csv(os.path.join(MAPPING_DIR, transcripts_f), index_col=0)
    tdf["global_z"] = tdf["global_z"].astype(int)
    tdf = tdf.drop(columns=["fov", "x", "y"])
    tdf = tdf[tdf["transcript_id"]!= '-1']
    A= pd.read_csv(os.path.join(MAPPING_DIR, transform_f),header=None,sep=' ').values
    XY = tdf[["global_x","global_y"]]
    aug_XY = np.hstack((XY,np.ones((XY.shape[0],1))))
    pixel_coords = (A @ np.transpose(aug_XY))[:-1] # Last row is dummy variable so it gets removed
    tdf[["p_x","p_y"]] = (np.transpose(pixel_coords))
    tdf[['p_x',"p_y"]] = ((tdf[['p_x','p_y']].values)//downsample_ratio).astype(int)

    
    img_new = np.zeros((img.shape[0],img.shape[1],3))
    img_new[:,:,0] = img.copy()
    
    # tdf['p_y'] = np.minimum(tdf['p_y'],img.shape[0]-1) #rounding error
    # tdf['p_x'] = np.minimum(tdf['p_x'],img.shape[1]-1) #rounding error
    tdf['layer']=1
    pixel_coords = tdf[['p_y','p_x','layer']].values
    top = np.percentile(img_new[:,:,0],97)
    img_new[tuple(pixel_coords.T)]=top
    del tdf,img
    gc.collect()
    return img_new #redundant

def make_maxz_mask():
    tic=time.time()
    max_img = None
    for z in args.zlayers:
        fname = os.path.join(IMAGES_DIR, f"mosaic_{args.channel}_z{z}.tif")
        assert os.path.exists(fname), f"Data file {fname} is not present for given channel: '{args.channel}' and z_stack: {z}"
        img = imread(fname)
        print("image shape: ", img.shape)

        if max_img is None:
            max_img = img
        else:
            max_img = np.maximum(max_img,img)
    if args.downsample_ratio != 1:
        max_img = cv2.resize(max_img,(max_img.shape[1]//args.downsample_ratio, max_img.shape[0]//args.downsample_ratio))
    print(max_img.shape)
    print(f"Segmenting max_z image with shape {max_img.shape}",flush=True)
    if args.augment_with_transcripts:
        max_img = _augment(img=max_img,downsample_ratio=args.downsample_ratio)
    if args.model_name is None:
        masks, _, _, _ = model.eval(
            max_img, diameter=args.diameter, channels=[[0,0]],
            flow_threshold =args.flow_threshold,cellprob_threshold =args.cellprob_threshold) #Outputs are normally: masks,flows,styles,diams
    else:
        masks, _, _, = model.eval(
            max_img, diameter=args.diameter, channels=[[0,0]],
            flow_threshold =args.flow_threshold,cellprob_threshold =args.cellprob_threshold) # Masks, flows, styles
            
    np.save(os.path.join(OUTPUT_DIR, f"{args.trial_name}_mosaic_maxz"), arr=masks, allow_pickle=True)
    toc=time.time()
    print(f"Finished creating segmentation file in {toc-tic} seconds",flush=True)
        

def make_single_mask(z):
    tic= time.time()
    fname = os.path.join(IMAGES_DIR, f"mosaic_{args.channel}_z{z}.tif")
    assert os.path.exists(fname), f"Data file {fname} is not present for given channel: '{args.channel}' and z_stack: {z}"
    img = imread(fname)
    if args.boundaries is not None:
        YL,YU,XL,XU=args.boundaries
        img = img[YL:YU,XL:XU]
    if args.downsample_ratio != 1:
        img = cv2.resize(img,(img.shape[1]//args.downsample_ratio, img.shape[0]//args.downsample_ratio))
    print(f"Segmenting {fname}",flush=True)
    if args.model_name is None:
        masks, _, _, _ = model.eval(
            img, diameter=args.diameter, channels=[[0,0]],
            flow_threshold =args.flow_threshold,cellprob_threshold =args.cellprob_threshold) #Outputs are normally: masks,flows,styles,diams
    else:
        masks, _, _, = model.eval(
            img, diameter=args.diameter, channels=[[0,0]],
            flow_threshold =args.flow_threshold,cellprob_threshold =args.cellprob_threshold) # Masks, flows, styles

    print(f"Number of cells in z-layer #{z}: {np.max(masks)}",flush=True)
    np.save(os.path.join(OUTPUT_DIR, f"{args.trial_name}_mosaic_{args.channel}_z{z}"), arr=masks, allow_pickle=True)
    toc=time.time()
    print(f"Finished segmenting {fname} and saved to disk in time {toc-tic}",flush=True)


def make_masks():
    tic= time.time()
    for z in args.zlayers:
        make_single_mask(z)
    toc=time.time()
    print(f"Total time to segment and save {len(args.zlayers)} image files = {toc-tic} seconds",flush=True)

def check_args(args):
    assert args.zlayers is not None
    if args.boundaries is not None:
        assert len(args.boundaries) == 4

def merge_masks(args):
    joint_masks = None
    tic = time.time()
    for z in sorted(args.zlayers):
        output_f = os.path.join(OUTPUT_DIR, f"{args.trial_name}_mosaic_{args.channel}_z{z}.npy")
        mask_z = np.load(output_f,allow_pickle=True)
        if joint_masks is None:
            joint_masks = np.zeros(( len(args.zlayers), mask_z.shape[0],mask_z.shape[1] ))
        joint_masks[z,:,:] = mask_z
    np.save(os.path.join(OUTPUT_DIR,f"masks_combined_{args.trial_name}.npy"),arr=joint_masks,allow_pickle=True)
    toc=time.time()
    print(f"Time taken to reload individual masks and put into one ndarray: {toc-tic}")

if __name__=="__main__":
    check_args(args)
    print(args.subdirectory)
    print(f"Diameter: {args.diameter}, flow_threshold: {args.flow_threshold}, cellprob_threshold: {args.cellprob_threshold}",flush=True)
    if args.resegment:
        print("Running sequential version # processes == 1", flush=True)
        if args.max_z:
            make_maxz_mask()
        else:
            make_masks()
    if args.merge and not args.max_z:
        print("Merging masks",flush=True)
        merge_masks(args)
    else:
        print("Merge=True ignored since maxz is being done",flush=True)
        

    