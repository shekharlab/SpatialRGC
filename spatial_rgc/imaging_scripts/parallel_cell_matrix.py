import pandas as pd
import numpy as np
import scanpy as sc
from cellpose import utils
import multiprocessing
from multiprocessing import Pool
import argparse
import os
import time

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--boundaries", help="Boundaries if running test",nargs='+',type=int, default=None)
parser.add_argument("-n", "--trial_name", help="Name of segmentation file",type=str)
parser.add_argument("-p", "--num_processes", type=int, default=1)
parser.add_argument("-s", "--downsample_ratio", help="downsample factor to use", type=int,default=1)
parser.add_argument("-S", "--subdirectory", type=str, default="")

parser.add_argument("-m", "--max_z", action='store_true', default=False)


args = parser.parse_args()
OUTPUT_DIR = os.path.join(BASE_DIR,"outputs", args.subdirectory)
MAPPING_DIR = os.path.join(BASE_DIR,"mappings",args.subdirectory)

if args.max_z:
    seg_file=os.path.join(OUTPUT_DIR, f"{args.trial_name}_mosaic_maxz.npy")
else:
    seg_file = os.path.join(OUTPUT_DIR, f"stitched_masks_{args.trial_name}_comb.npy")
    
SEG_M = np.load(seg_file,allow_pickle=True).astype(int)

def get_sizes_centroids(cell_id,run2D):
    if run2D:
        coord_y,coord_x= np.array(np.where(SEG_M==cell_id))
    else:
        coord_z,coord_y,coord_x = np.array(np.where(SEG_M==cell_id))
        
    # print(f"Cell_id: {cell_id} done",flush=True)
    
    if run2D:
        max_area = len(coord_y)
    else:
        max_area = np.max(np.bincount(coord_z, minlength=SEG_M.shape[0]))
    return (len(coord_x), np.mean(coord_y), np.mean(coord_x),max_area)  
    
def make_matrix_parallel(base_dir,trial_name, boundaries, downsample_ratio, num_processes=1, z_thick = 1.5, transcripts_f="detected_transcripts.csv",transform_f="micron_to_mosaic_pixel_transform.csv"):
    test=False
    if boundaries is not None:
        YL,YU,XL,XU = boundaries
        test=True
    
    run2D= (len(SEG_M.shape) == 2)
    
    tdf = pd.read_csv(os.path.join(MAPPING_DIR, transcripts_f), index_col=0)
    tdf["global_z"] = tdf["global_z"].astype(int)
    tdf = tdf.drop(columns=["fov", "x", "y"])
    tdf = tdf[tdf["transcript_id"]!= '-1']

    A= pd.read_csv(os.path.join(MAPPING_DIR, transform_f),header=None,sep=' ').values
    XY = tdf[["global_x","global_y"]]
    aug_XY = np.hstack((XY,np.ones((XY.shape[0],1))))
    pixel_coords = (A @ np.transpose(aug_XY))[:-1] # Last row is dummy variable so it gets removed
    tdf[["p_x","p_y"]] = (np.transpose(pixel_coords))

    if test:
        filter = ((tdf["p_x"]>= XL) & (tdf["p_x"] < XU) & (tdf["p_y"]>= YL) & (tdf["p_y"] < YU))
        tdf = tdf[filter]
        tdf["p_x"] = tdf["p_x"] - XL
        tdf["p_y"] = tdf["p_y"] - YL
    
    tdf[['p_x',"p_y"]] = ((tdf[['p_x','p_y']].values)//downsample_ratio).astype(int)

    print("Assigning transcripts cell_ids",flush=True)
    if run2D:
        pixel_coords = tdf[["p_y","p_x"]].values
    elif len(SEG_M.shape) ==3:
        pixel_coords = tdf[["global_z","p_y","p_x"]].values
    else:
        assert AssertionError(f"Segmented mask is neither 2D nor 3D and has shape {SEG_M.shape} instead")
    tdf["cell_id"] = SEG_M[tuple(pixel_coords.T)]
    tdf.to_csv(os.path.join(OUTPUT_DIR, f"transcripts_mapped_{trial_name}.csv"),index=False)
    tdf = tdf[tdf["cell_id"] != 0]
    cell_by_gene = sc.AnnData(tdf.groupby(by="cell_id",sort=True)["gene"].value_counts().unstack().fillna(0))
    cell_by_gene.obs.index =cell_by_gene.obs.index.astype(int)
    print("Number of cells with non zero # of transcripts: ", len(cell_by_gene.obs.index))

    # Calculate cell areas/volumes and centroids in space for the cells that have AT LEAST 1 transcript
    # SIZE REFERS TO VOLUME FOR 3D SEGMENTATION, 2D FOR AREA
    print("Calculating areas and volumes",flush=True)
    tic=time.time()
    args= tuple( zip( list(cell_by_gene.obs.index), np.repeat(run2D, len(cell_by_gene.obs.index) ) ) )
    with Pool(processes=num_processes) as p:
        outputs= np.array(p.starmap(get_sizes_centroids,args))
    abs_jac = np.abs(np.linalg.det(A[:-1,:-1]))
    print("Jacobian determinant: (before correcting for downsampling) ", abs_jac)
    sizes = outputs[:,0]*(downsample_ratio**2)/abs_jac
    centroids_y = (outputs[:,1]*downsample_ratio - A[1,2])/A[1,1]
    centroids_x = (outputs[:,2]*downsample_ratio - A[0,2])/A[0,0]
    max_areas = outputs[:,3]*(downsample_ratio**2)/abs_jac
    toc=time.time()
    print(f"Time to calculate areas and volumes: {toc-tic}", flush=True)
    if not run2D:
        sizes = np.array(sizes)*z_thick
    
    cell_by_gene.obs["Size"] = sizes 
    cell_by_gene.obs["center_x"] = centroids_x
    cell_by_gene.obs["center_y"] = centroids_y
    cell_by_gene.obs["max_areas"] = max_areas
    
    if not run2D:
        fname= os.path.join(OUTPUT_DIR, f"Cellpose_model_{trial_name}.h5ad")
    else:
        fname= os.path.join(OUTPUT_DIR, f"Cellpose_model_maxz_{trial_name}.h5ad")
    sc.AnnData.write_h5ad(cell_by_gene, filename= fname)
    print(f"Finished saving anndata to: {fname}")

if __name__=="__main__":
    make_matrix_parallel(BASE_DIR, args.trial_name, args.boundaries, args.downsample_ratio, num_processes=args.num_processes, z_thick = 1.5, transcripts_f="detected_transcripts.csv",transform_f="micron_to_mosaic_pixel_transform.csv")
    