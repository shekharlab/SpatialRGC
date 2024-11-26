import argparse
import datetime
import gc
import logging
import numpy as np
import os
import sys
import time

import matplotlib.pyplot as plt
import copy

from cellpose import io
from cellpose import models
from cellpose.io import imread
from multiprocessing import Pool,Process
from multiprocessing.shared_memory import SharedMemory


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stitch_threshold", help="Run stitch and threshold to use",type=float, default=0)
parser.add_argument("-z", "--zlayers", help="Z_layers to use", nargs='+', type=int, default=[0,1,2,3,4,5,6])
parser.add_argument("-n", "--trial_name", help="Name of segmentation file",type=str)
parser.add_argument("-p", "--num_processes", type=int, default=1)
parser.add_argument("-S", "--subdirectory", type=str, default="")


args = parser.parse_args()
STITCH_THRESHOLD = args.stitch_threshold
OUTPUT_DIR = os.path.join(BASE_DIR,"outputs", args.subdirectory)

# Masks has shape (z,Y,X)
masks = np.load(os.path.join(OUTPUT_DIR, f"masks_combined_{args.trial_name}.npy"), allow_pickle=True)
comb = np.zeros(masks.shape)


def max_overlap_ids(indices_size, indices_shape,indices_dtype,z,start,stop):
    mask = masks[z,:,:]
    shm=SharedMemory(size=mask.nbytes, name=f"pm_{args.trial_name}_{z}")
    prev_mask = np.ndarray(shape=mask.shape, dtype=mask.dtype, buffer=shm.buf)

    shm_indices = SharedMemory(size=indices_size,name=f"i_{args.trial_name}_{z}")
    all_ids = np.ndarray(shape=indices_shape, dtype=indices_dtype, buffer=shm_indices.buf)
    new_id_set = np.arange(start,stop)+1
    for new_id in new_id_set:
        upper_mask = (mask==new_id)
        inters_cell_ids = list(np.unique(prev_mask[upper_mask]))
        try:
            inters_cell_ids.remove(0.)
        except:
            pass
        jaccard_lst = []
        for prev_id in inters_cell_ids:
            prev_id = int(prev_id)
            lower_mask = (prev_mask == prev_id)
            overlap = np.sum(lower_mask & upper_mask)
            union = np.sum(lower_mask | upper_mask)
            jaccard_idx = overlap/union
            jaccard_lst.append(jaccard_idx)
        max_id = None
        if (len(inters_cell_ids) == 0) or (jaccard_lst[np.argmax(jaccard_lst)] <= STITCH_THRESHOLD):# record that we need a new cell and assign it a new id (since ids are 1 to 45 for each) 
            max_id = -1
        else: # Reassign cell to previous cell with maximnum overlap if above threshold
            max_id = inters_cell_ids[np.argmax(jaccard_lst)]
        all_ids[new_id-1] = max_id
    shm.close()
    shm_indices.close()


def replace_job(indices_size, indices_shape,indices_dtype,z,start,stop):
    shm = SharedMemory(name=f"nm_{args.trial_name}_{z}")
    new_mask = np.ndarray(shape=masks[z].shape,dtype=masks[z].dtype, buffer=shm.buf)
    shm_indices = SharedMemory(size=indices_size,name=f"i_{args.trial_name}_{z}")
    all_ids = np.ndarray(shape=indices_shape, dtype=indices_dtype, buffer=shm_indices.buf)
    for key in np.arange(start=start+1,stop=stop+1):
        new_mask[masks[z,:,:]==key] = all_ids[key-1]
    shm.close()
    shm_indices.close()
        
def _partitionIndexes(totalsize, numberofpartitions):
    # Compute the chunk size (integer division)
    chunksize = totalsize // numberofpartitions
    # How many chunks need an extra 1 added to the size?
    remainder = totalsize - chunksize * numberofpartitions
    a = 0
    for i in range(numberofpartitions):
        b = a + chunksize + (i < remainder)
        # Yield the inclusive-inclusive range
        yield (a, b - 1)
        a = b

"""RETURNS NOTHING, JUST UPDATES GLOBAL VARIABLE (the matrix 'comb')"""
def parallel_stitch(z):
    # MAKE THESE VARIABLES WRITEABLE
    global NUM_CELLS
    global comb
    if z ==0:
        comb[z,:,:]=masks[z]
        NUM_CELLS=np.max(masks[z])
        print(f"Num cells: {NUM_CELLS}")
        return
    mask = masks[z,:,:]
    new_cell_ids = np.arange(start=1, stop=np.max(mask)+1) # Returns 1....cell_id
    print(f"Z is {z}. There are {len(new_cell_ids)} cells segmented in this layer. Beginning to find which ids to replace.",flush=True)
    tic = time.time()
    # Buffer for old mask
    shm=SharedMemory(create=True, size=mask.nbytes, name=f"pm_{args.trial_name}_{z}")
    old_mask = np.ndarray(shape = mask.shape, dtype=mask.dtype, buffer=shm.buf)
    old_mask[:] = comb[z-1,:,:]
    # Buffer, where index i will have the entry for which id new_id i maps to
    shm_indices=SharedMemory(create=True, size=new_cell_ids.nbytes, name=f"i_{args.trial_name}_{z}")
    output_ids = np.ndarray(shape=new_cell_ids.shape,dtype=new_cell_ids.dtype,buffer=shm_indices.buf)
    output_ids[:] = 0
    num_processes = args.num_processes
    processes=[]
    for start,stop in _partitionIndexes(len(new_cell_ids),num_processes):
        p=Process(target=max_overlap_ids, args=(new_cell_ids.nbytes,new_cell_ids.shape, new_cell_ids.dtype,z,start,stop))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    toc = time.time()
    shm.close()
    shm.unlink()
    print(f"Time to determine which id's to replace for layer {z}: {toc-tic} seconds",flush=True)
    
    tic = time.time()
    to_add = output_ids==-1
    n_new = len(output_ids[to_add])
    print(n_new)
    output_ids[to_add] = np.arange(n_new)+(NUM_CELLS+1)
    NUM_CELLS+=n_new

    shm=SharedMemory(create=True, size=mask.nbytes, name=f"nm_{args.trial_name}_{z}")
    new_mask = np.ndarray(shape= mask.shape, dtype=mask.dtype, buffer=shm.buf)
    new_mask[:] = 0
    MAP_DICT = dict(zip(new_cell_ids, output_ids))
    processes=[]
    for start,stop in _partitionIndexes(len(new_cell_ids),num_processes):
        p=Process(target=replace_job, args=(output_ids.nbytes, output_ids.shape,output_ids.dtype,z,start,stop))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    comb[z,:,:] = new_mask.copy()
    shm.close()
    shm.unlink()
    shm_indices.close()
    shm_indices.unlink()
    toc = time.time()
    print(f"Num cells after merging layer {z}: {NUM_CELLS}",flush=True)
    print(f"Time to reassign cell_id's for layer {z}: {toc-tic}, seconds",flush=True)
    return output_ids

def check_args(args):
    assert args.trial_name is not None
    for z in args.zlayers:
        try:
            SharedMemory(size=masks[0].nbytes, name=f"pm_{args.trial_name}_{z}").unlink()
        except:
            pass
        try:
            SharedMemory(size=masks[0].nbytes, name=f"i_{args.trial_name}_{z}").unlink()
        except:
            pass
        try:
            SharedMemory(size=masks[0].nbytes, name=f"nm_{args.trial_name}_{z}").unlink()
        except:
            pass

if __name__ == "__main__":
    check_args(args)
    print(f"Stitch threshold: {args.stitch_threshold}")
    for z in args.zlayers:
        print(f"Stitching layer {z}")
        parallel_stitch(z)
    output_f = os.path.join(OUTPUT_DIR, f"stitched_masks_{args.trial_name}_comb")
    print(f"Saving stitched matrix: {output_f}",flush=True)
    np.save(output_f,arr=comb,allow_pickle=True)
    print("Finished stitching",flush=True)
    print("-------------------",flush=True)



