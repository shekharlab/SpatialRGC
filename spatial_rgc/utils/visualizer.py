from matplotlib.collections import PatchCollection
import os
import h5py
import numpy as np 
import pandas as pd
import tifffile
import cv2
import matplotlib.pyplot as plt

from cellpose import models
from cellpose.io import imread
from cellpose import io,utils,plot
import cv2
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib as mpl
import seaborn as sns

import spatial_rgc.utils.segmenter as se

from collections import defaultdict



# FNAMES = os.listdir(os.path.join(BASE_DIR,"cell_boundaries"))
# FULL_PATH_NAMES = list(map(lambda x: os.path.join(BASE_DIR,"cell_boundaries",x), FNAMES))
# TRANSFORM_F = f"mappings/{subdirectory}/micron_to_mosaic_pixel_transform.csv"
# TRANSCRIPTS_F=f"detected_transcripts.csv"
# CELLPOSE_DIR = os.path.join(BASE_DIR,"training_examples")


def micron_to_mosaic_fxn(A, xy_micron,XL=0,YL=0,dr=None):
    aug_m = np.vstack( (xy_micron, np.ones((1,xy_micron.shape[1])) ) )
    xy_mosaic  = (A@aug_m)[:-1]
    shift_arr = np.array([[XL]*xy_micron.shape[1], [YL]*xy_micron.shape[1]])
    xy_mosaic  = xy_mosaic - shift_arr
    if (dr is not None) and (dr != 1):
        xy_mosaic = xy_mosaic//dr
    xy_mosaic=xy_mosaic.astype(int)
    return xy_mosaic
    #xy micron: rows1=x,row2=y

class Visualizer():
    # Micron regions = [ [[XL1,XU1],[YL1,YU1]]...]
    CMAP_CLASS = plt.get_cmap('Paired')
    CLASS_COLOR_MAP = {"RGC":CMAP_CLASS(3),"Amacrine":CMAP_CLASS(7), "Other":CMAP_CLASS(9)}
    FINE_DPI = 200
    EXTRA_FINE_DPI = 300
   

    def __init__(self, micron_regions, stitched_m,downsample_ratio,image_script_dir, subdirectory,trial_name="c2knl_full_retina"):
        transform_f = os.path.join(image_script_dir,"mappings",subdirectory,"micron_to_mosaic_pixel_transform.csv")
        num_regions = len(micron_regions)
        mosaic_regions = []
        self.A =pd.read_csv(transform_f, header=None, sep=' ').values
        for micron_region in micron_regions:
            mosaic_region = micron_to_mosaic_fxn(self.A,micron_region,dr=4) # also handles downsampled coordinates
            mosaic_regions.append(mosaic_region)
        
        self.transcripts_f=os.path.join(image_script_dir,"mappings",subdirectory, "detected_transcripts.csv")
        self.transcripts_assigned_f = os.path.join("data",subdirectory,f"transcripts_mapped_{trial_name}.csv")
        self.micron_regions = micron_regions
        self.mosaic_regions = mosaic_regions
        self.image_script_dir = image_script_dir
        self.dr = downsample_ratio
        self.stitched_m = stitched_m
        self.subdirectory=subdirectory
        self.figure_dir = os.path.join(image_script_dir,"..","figures",subdirectory)
    
    def show_all_transcripts(self,s=0.00001):
        tdf = pd.read_csv(self.transcripts_f)
        plt.scatter(tdf['global_x'],tdf['global_y'],s=s)
        plt.grid()
        plt.show()
        return tdf
            
    
    def visualize_area(self,z,r_n,plot_masks=False,stain="Cellbound2",axis_fontsize=24, origin='lower',train_sample=None,vmin=None,cmap='Greys',micron_input=True,vmax=None,adjust_axes=False,invert_yaxis=True,fname=""):
        plot_height=10
         #In micron coordinates (in x,y)
        if micron_input:
            XL_m,XU_m,YL_m,YU_m = self.micron_regions[r_n].reshape(-1)
            XL,XU,YL,YU = self.mosaic_regions[r_n].reshape(-1)
        else:
            assert isinstance(r_n,np.ndarray)
            XL,XU,YL,YU = r_n.reshape(-1)
        print(f"Pixel coords: x=({XL},{XU}), y=({YL},{YU})")
        n_cols=3
        fig,ax=plt.subplots(ncols=n_cols,nrows=1,figsize=(plot_height*4/3*n_cols, plot_height))
        mpl.rcParams['font.sans-serif'] = "Arial"
        mpl.rcParams['font.family'] = "sans-serif"
        mpl.rcParams.update({'font.size': 24})
        # Load,resize, and create window on image and segmentation matrix
        img,z_l = self._z_window(z,r_n,stain=stain,micron_input=False)
        #print(f"Minimum intensity: {self.min_intensity}, Maximum intensity:{self.max_intensity}")
        print(np.min(img),np.max(img))

        if plot_masks:
            outlines = utils.outlines_list(z_l)
            for o in outlines:
                ax[0].plot(o[:,0],o[:,1],color='r',linewidth=1)
        for ax_i in ax:
            ax_i.imshow(img,origin=origin,vmin=vmin,vmax=vmax,cmap=cmap)
        if train_sample is not None:
            cv2.imwrite(train_sample,img)
        
        if adjust_axes:
            self._adjust_axes(img.shape, ax,r_n,axis_fontsize,adjust_bounds=True,invert_yaxis=invert_yaxis)

        if fname is None:
            fname = os.path.join(self.figure_dir,f"{stain}_z{z}_r{r_n}")
        else:
            fname = os.path.join(self.figure_dir,fname)

           
        plt.savefig(fname,bbox_inches='tight',dpi=200)
        print("Saved image to:", fname)

        plt.show()
        return img
    
        
    
    def _z_window(self,z,r_n,img=None,micron_input=True,stain="Cellbound2"):
        if isinstance(r_n,int):
            XL,XU,YL,YU = self.mosaic_regions[r_n].reshape(-1)
        elif isinstance(r_n,np.ndarray):#overloaded
            if micron_input:
                XL,XU,YL,YU = micron_to_mosaic_fxn(self.A,r_n,dr=self.dr).reshape(-1) # also handles downsampled coordinates
            else:
                XL,XU,YL,YU = r_n.reshape(-1)
        img_file=f"mosaic_{stain}_z{z}_dr{self.dr}.tif"
        img_path = os.path.join(self.image_script_dir,"images", self.subdirectory, img_file)


        if img is None:
            img = imread(img_path)
            img = img[YL:YU,XL:XU]

        if self.stitched_m is not None:
            if len(self.stitched_m.shape) == 2:
                z_l= self.stitched_m[YL:YU,XL:XU]
            else:
                z_l = self.stitched_m[z][YL:YU,XL:XU].copy()
        else:
            print("Warning: No stitched file loaded")
            z_l=None
        return img,z_l

    def check_cell_ids(self,r_n,plot_masks=False):
        XL,XU,YL,YU = self.mosaic_regions[r_n].reshape(-1)
        for z in range(self.stitched_m.shape[0]):
            print(np.unique(self.stitched_m[:, YL:YU,XL:XU]))
    
    #TODO: Create local version
    def _calc_avg_intensity(ids, mask_mat, img):
        intensity = defaultdict(lambda x: 0)
        n_counts = defaultdict(lambda x: 0)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                id = img[y,x]
                intensity[n_counts]


    def _adjust_axes(self,img_shape, ax,r_n,axis_fontsize,adjust_bounds=True,num=5,tickpad=10,invert_yaxis=False):
        XL_m,XU_m,YL_m,YU_m = self.micron_regions[r_n].reshape(-1)
        for i in range(ax.shape[0]):
            if adjust_bounds:
                ax[i].set_ylim(0-0.5,img_shape[0]-0.5)
                ax[i].set_xlim(0-0.5,img_shape[1]-0.5)
            ax[i].set_xlabel(r'x ($\mu m$)',fontsize=axis_fontsize)
            ax[i].set_ylabel(r'y ($\mu m$)',fontsize=axis_fontsize)
            ax[i].set_xticks(np.linspace(start=0-0.5,stop=img_shape[1]-0.5,num=num))
            ax[i].set_yticks(np.linspace(start=0-0.5,stop=img_shape[0]-0.5,num=num))

            ax[i].set_xticklabels(np.linspace(start=XL_m,stop=XU_m,num=num),fontsize=axis_fontsize)
            ax[i].set_yticklabels(np.linspace(start=YL_m,stop=YU_m,num=num),fontsize=axis_fontsize)
            ax[i].tick_params(axis='both',which='both',pad=tickpad)
            if invert_yaxis:
                ax[i].invert_yaxis()
    
    
    # Return transcripts datafarme containing specific genes, using downsampled mosaic coordinates
    def _get_transcripts_df(self, r_n,gene_groups=None, z_t=None,s=0.1):
        tdf = pd.read_csv(self.transcripts_assigned_f)
        tdf_g = tdf
        all_genes = []
        if gene_groups is not None:
            for group in gene_groups:
                genes = gene_groups[group]
                all_genes = all_genes + genes
            tdf_g = tdf_g[tdf_g['gene'].isin(all_genes)]
        else:
            all_genes = [x for x in np.unique(tdf['gene']) if 'Blank' not in x]
            tdf_g = tdf_g[tdf_g['gene'].isin(all_genes)]
        if z_t is not None:
            tdf_g = tdf_g[tdf_g['global_z'].isin(all_genes)]
        
        if isinstance(r_n,int):
            XL,XU,YL,YU = self.mosaic_regions[r_n].reshape(-1)
        elif isinstance(r_n,np.ndarray):#overloaded
            XL,XU,YL,YU = micron_to_mosaic_fxn(self.A,r_n,dr=self.dr).reshape(-1) # also handles downsampled coordinates

        tdf_g = tdf_g[(tdf_g['p_x'] >=XL) & (tdf_g['p_x'] <= XU) & (tdf_g['p_y'] >=YL) & (tdf_g['p_y'] <= YU)]
        tdf_g[['p_x','p_y']] = tdf_g[['p_x','p_y']].values - (XL,YL)
        return tdf_g
        
    def _plot_transcripts(self, ax, r_n,gene_groups=None, z_t=None,s=0.1,tdf_g=None,color_groups=None):                        
        if tdf_g is None:
            tdf_g = self._get_transcripts_df(r_n,gene_groups=gene_groups, z_t=None,s=0.1)
        if gene_groups is not None:
            for group in gene_groups:
                genes= gene_groups[group]
                tdf_gg = tdf_g[tdf_g['gene'].isin(genes)]
                if color_groups is not None:
                    color = color_groups[group]
                else:
                    color=None

                ax.scatter(tdf_gg['p_x'],tdf_gg['p_y'],label=group,s=s,rasterized=True,color=color)
    
        else:
            sns.scatterplot(data=tdf_g,x='p_x',y='p_y',color='orange',s=s,label='mRNA',ax=ax)
            # ax.scatter(tdf_g['p_x'],tdf_g['p_y'],label="mRNA",s=s,rasterized=True)
        ax.legend(bbox_to_anchor=(1.0,1.0),markerscale=5, prop={'size':36})
        return tdf_g
        
        
    # Plots outlines and stitching (stitching if remaining_z); caches outliesn adn outlines_stitch_lst
    def _plot_outlines(self,ax,z_l,r_n, remaining_z, masks_drawn=None, outlines=None,outlines_stitch_lst=None, stain="Cellbound2",color='red',stitch_color='blue',linewidth=1):
        if masks_drawn is None:
            masks_drawn = set()

        if outlines is None:
            z_filt,masks_drawn = self._filter_ids(z_l,masks_drawn)
            outlines=utils.outlines_list(z_filt)
            masks_drawn  = masks_drawn.union(set(np.unique(z_l)))


        for o in outlines:
            ax.plot(o[:,0],o[:,1],color=color,linewidth=linewidth)
            

        if outlines_stitch_lst is None or len(outlines_stitch_lst)==0:
            outlines_stitch_lst = []
            for zr in remaining_z:
                _, z_l = self._z_window(zr,r_n,stain=stain,img='dummy')
                z_filt,masks_drawn = self._filter_ids(z_l,masks_drawn)
                outlines_stitch_lst.append(utils.outlines_list(z_filt))
        for outlines_stitch in outlines_stitch_lst:
            for o in outlines_stitch:
                ax.plot(o[:,0],o[:,1],color=stitch_color,linewidth=linewidth)

        return outlines,outlines_stitch_lst,masks_drawn

    def visualize_transcripts(self,z,r_n,gene_groups,remaining_z, z_t=None,plot_masks=False,group_name=None, cmap='Greys',plot_height=10,axis_fontsize=24,linewidth=1,s=3,color_groups=None,fname=None,vmin=None,vmax=None,orig_seg_color='red',stitch_color='blue',stain="Cellbound2",invert_yaxis=False,suffix="png"):
        mpl.rcParams['font.sans-serif'] = "Arial"
        mpl.rcParams['font.family'] = "sans-serif"
        img,z_l = self._z_window(z,r_n,stain=stain)
        ncols = 3
        fig,ax=plt.subplots(ncols=ncols,nrows=1,figsize=(plot_height*ncols*4/3, plot_height))
        ax[0].imshow(img,cmap=cmap,vmin=vmin,vmax=vmax)
        ax[1].imshow(img,cmap=cmap,vmin=vmin,vmax=vmax)
        #img=np.zeros(img.shape)
        ax[2].imshow(img,cmap=cmap,vmin=vmin,vmax=vmax)
        outlines,outlines_stitch_lst=None,None
        if plot_masks:
            for i in range(1,3):
                outlines, outlines_stitch_lst,_ = self._plot_outlines(ax[i],z_l,r_n, remaining_z=remaining_z, outlines=outlines,outlines_stitch_lst=outlines_stitch_lst, color=orig_seg_color,stitch_color=stitch_color,linewidth=linewidth)
        # self._plot_transcripts(ax=ax[1],r_n=r_n,gene_groups=gene_groups,z_t=z_t,s=s)
        self._plot_transcripts(ax=ax[2],r_n=r_n,gene_groups=gene_groups,z_t=z_t,s=s,color_groups=color_groups)
        self._adjust_axes(img.shape, ax,r_n,axis_fontsize,adjust_bounds=True,invert_yaxis=invert_yaxis)
        if fname is None:
            if group_name is None:
                fname = os.path.join(self.figure_dir,f"TEST: transcripts_z{z}_r{r_n}.{suffix}")
            else:
                fname = os.path.join(self.figure_dir, f"TEST: {group_name}_showcase_z{z}_r{r_n}.{suffix}")
        else:
            fname = os.path.join(self.figure_dir,fname)

           
        fig.savefig(fname,bbox_inches='tight',dpi=200)
        print("Saved image to:", fname)
    
    def test_segmentation(self,z,r_n,model_group,subtype,plot_height=10,axis_fontsize=16, stain="DAPI",color='red',fname=None, t_size=0.005, train_dir=os.path.join("imaging_scripts","training_samples","SC_training"),invert_yaxis=True,masks=None, **kwargs):
        img,z_l = self._z_window(z,r_n,stain=stain)
        ncols=4
        fig,ax=plt.subplots(ncols=ncols,nrows=1,figsize=(plot_height*ncols*4/3, plot_height))
        ax[0].imshow(img,cmap='Greys')
        blank_mat =  np.zeros((img.shape[0],img.shape[1],3))
        blank_mat[:,:] = (1,1,1)
        ax[1].imshow(blank_mat)
        ax[2].imshow(blank_mat)
        ax[3].imshow(img,cmap='Greys')
        self._adjust_axes(img.shape, ax,r_n,axis_fontsize,adjust_bounds=True,invert_yaxis=invert_yaxis)
        s = se.Segmenter(name="sample",models_dir=os.path.join("imaging_scripts","models"))
        tdf_g = self._plot_transcripts(ax=ax[1],r_n=r_n,gene_groups=None,z_t=z,s=t_size)
        tdf_g = self._plot_transcripts(ax=ax[2],r_n=r_n,gene_groups=None,z_t=z,s=t_size,tdf_g=tdf_g)
        tdf_g = self._plot_transcripts(ax=ax[3],r_n=r_n,gene_groups=None,z_t=z,s=t_size)
        if masks is None:# CHANGE THIS FOR BAYSOR WITH PRIOR SPECIFICALLY
            masks=s.run_segmentation(img,**kwargs, model_group=model_group,subtype=subtype,tdf=tdf_g)
        if masks is not None:
            outlines, outlines_stitch_lst,_ = self._plot_outlines(ax[2],masks, r_n, remaining_z=[], outlines=None,outlines_stitch_lst=None, color=color,stitch_color='blue',linewidth=1)
            self._plot_outlines(ax[3],masks, r_n, remaining_z=[], outlines=outlines,outlines_stitch_lst=None, color='red',stitch_color='blue',linewidth=1)
            tdf_g['p_y'] = np.minimum(tdf_g['p_y'],masks.shape[0]-1) #rounding error
            tdf_g['p_x'] = np.minimum(tdf_g['p_x'],masks.shape[1]-1) #rounding error
            pixel_coords = tdf_g[['p_y','p_x']].values
            tdf_g["cell_id"] = masks[tuple(pixel_coords.T)]
        

        if fname is None:
            fname=f"{model_group}_{subtype}_{r_n}.png"

        output=os.path.join(self.figure_dir,fname)
        fig.savefig(output,dpi=100)
        print("Saving figure to ", output)
        return masks,tdf_g
        




    # MODIFIES z_l by mkaing a copy
    def _filter_ids(self,z_l,masks_drawn):
        z = z_l.copy()
        for id in masks_drawn:
            z[z==id] = 0
        masks_drawn = masks_drawn.union(set(np.unique(z)))
        return z,masks_drawn
    
    def threshold_area(self,z,r_n,stain='Cellbound3',threshold=5000,max_val=10000, plot_height=10):
        img,z_l = self._z_window(z,r_n,stain=stain)
        ncols=2
        thresh1 = np.where(img<threshold,300, max_val)
        fig,ax=plt.subplots(ncols=ncols,nrows=1,figsize=(plot_height*ncols*4/3, plot_height))
        ax[0].imshow(img)
        ax[1].imshow(thresh1)
        return img,thresh1


    
    def visualize_costain(self,adata,z,r_n,stain1='Cellbound2',stain2='Cellbound3',threshold=3000,plot_height=15,axis_fontsize=16,linewidth=1):
        mpl.rcParams['font.sans-serif'] = "Arial"
        mpl.rcParams['font.family'] = "sans-serif"
        img,z_l = self._z_window(z,r_n,stain=stain1)
        ncols = 3
        fig,ax=plt.subplots(ncols=ncols,nrows=1,figsize=(plot_height*ncols*4/3, plot_height))
        ax[0].imshow(img,cmap='Greys')
        self._plot_outlines(ax[0],z_l,r_n, remaining_z=[], masks_drawn=None, outlines=None,outlines_stitch_lst=None, color='red',stitch_color='blue',linewidth=linewidth)

        img,z_l = self._z_window(z,r_n,stain=stain2)
        ax[1].imshow(img)
        ax[2].imshow(img)
        not_marked = set(adata.obs[adata.obs[stain2] <= threshold].index.astype(int))
        z_l, not_marked = self._filter_ids(z_l,not_marked)
        self._plot_outlines(ax[2],z_l,r_n, remaining_z=[], masks_drawn=not_marked, outlines=None,outlines_stitch_lst=None, color='red',stitch_color='blue',linewidth=linewidth)
        self._adjust_axes(img.shape, ax,r_n,axis_fontsize,adjust_bounds=True)
        plt.savefig(os.path.join(self.figure_dir,f"{stain1}_{stain2}_z{z}_r{r_n}.png"),bbox_inches='tight')
        plt.show()
    
    def stain_histogram(self,adata,stain='Cellbound3',color='skyblue',fontsize=16):
        fig,ax = plt.subplots()
        mpl.rcParams['font.sans-serif'] = "Arial"
        mpl.rcParams['font.family'] = "sans-serif"
        ax.hist(adata.obs[stain],color=color,edgecolor='black')
        ax.set_xlabel("Average intensity of cell",fontsize=fontsize)
        ax.set_ylabel("Frequency",fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(os.path.join(self.figure_dir,f"Histogram_{stain}.png"),bbox_inches='tight')
        plt.show()



    def visualize_stitching(self, z, r_n, remaining_z, cmap='Greys',stain='Cellbound2',id_filter=None, plot_height=10,axis_fontsize=16,linewidth=1,axis_off=False,fname=None):
        mpl.rcParams['font.sans-serif'] = "Arial"
        mpl.rcParams['font.family'] = "sans-serif"
        img,z_l = self._z_window(z,r_n,stain=stain)
        n_cols=3
        fig,ax=plt.subplots(ncols=n_cols,nrows=1,figsize=(plot_height*4/3*n_cols, plot_height))
        ax[0].imshow(img,cmap=cmap)
        ax[1].imshow(img,cmap=cmap)
        ax[2].imshow(img,cmap=cmap)
       
        outlines,outlines_stitch_lst,masks_drawn=None,None,None
        if id_filter is not None:
            masks_drawn = id_filter
        outlines, outlines_stitch_lst,masks_drawn = self._plot_outlines(ax[1],z_l,r_n, remaining_z=[], masks_drawn=masks_drawn, outlines=outlines,outlines_stitch_lst=outlines_stitch_lst, color='red',stitch_color='blue',linewidth=1)
        outlines, outlines_stitch_lst,masks_drawn = self._plot_outlines(ax[2],z_l,r_n, remaining_z=remaining_z, masks_drawn=masks_drawn, outlines=outlines,outlines_stitch_lst=outlines_stitch_lst, color='red',stitch_color='blue',linewidth=1)

        self._adjust_axes(img.shape, ax,r_n,axis_fontsize,adjust_bounds=True)
        ax[0].set_title("Raw image",fontsize=24)
        ax[1].set_title("1 z-layer segmentation",fontsize=24)
        ax[2].set_title(f"{len(remaining_z)+1} z-layers stitched",fontsize=24)
        if axis_off:
            ax[2].set_axis_off()
        if fname is None:
            output_f= os.path.join(self.figure_dir, f"TEST: filt_stitching_showcase_z{z}_r{r_n}.png")
        else:
            output_f=os.path.join(self.figure_dir,fname)
        plt.savefig(output_f,bbox_inches='tight',dpi=200,rasterized=False)
        plt.show()
    
    def visualize_stitched_z(self,all_z, select_z,r_n, cmap='Greys',stain='Cellbound2',stitch_palette={7:'red',6:'orange',5:'yellow',4:'green',3:'blue',2:'pink',1:'purple',0:'grey'},id_filter=None, plot_height=10,axis_fontsize=16,linewidth=1,axis_off=False,fname=None):
        mpl.rcParams['font.sans-serif'] = "Arial"
        mpl.rcParams['font.family'] = "sans-serif"
        img,z_l = self._z_window(select_z,r_n,stain=stain)
        n_cols=3
        fig,ax=plt.subplots(ncols=n_cols,nrows=1,figsize=(plot_height*4/3*n_cols, plot_height))
        ax[0].imshow(img,cmap=cmap)
        ax[1].imshow(img,cmap=cmap)
        ax[2].imshow(img,cmap=cmap)
       
        outlines,outlines_stitch_lst,masks_drawn=None,None,None
        if id_filter is not None:
            masks_drawn = id_filter
        for z in all_z:
            _,z_l = self._z_window(z,r_n,stain=stain,img='dummy')
            outlines, outlines_stitch_lst,masks_drawn = self._plot_outlines(ax[2],z_l,r_n, remaining_z=[], masks_drawn=masks_drawn, outlines=None,outlines_stitch_lst=None, color=stitch_palette[z],stitch_color='blue',linewidth=1)

        self._adjust_axes(img.shape, ax,r_n,axis_fontsize,adjust_bounds=True)
        ax[0].set_title("Raw image",fontsize=24)
        ax[1].set_title("1 z-layer segmentation",fontsize=24)
        ax[2].set_title(f"{len(all_z)} z-layers stitched",fontsize=24)
        if axis_off:
            ax[2].set_axis_off()
        if fname is None:
            output_f= os.path.join(self.figure_dir, f"TEST: filt_stitching_showcase_z{z}_r{r_n}.png")
        else:
            output_f=os.path.join(self.figure_dir,fname)
        plt.savefig(output_f,bbox_inches='tight',dpi=200,rasterized=False)
        plt.show()
    
    def _plot_masks(self,ax,z_l,id_to_color,cmap,id_postfix,mask_mat=None):
        colored_ids = []
        if mask_mat is None:
            mask_mat = np.zeros((z_l.shape[0],z_l.shape[1],3))
            mask_mat[:,:] = (1,1,1)
            for y in range(mask_mat.shape[0]):
                for x in range(mask_mat.shape[1]):
                    c_id = str(int(z_l[y,x]))
                    c_id = f"{c_id}_{id_postfix}"
                    if c_id in id_to_color:
                        color = cmap(id_to_color[c_id])
                        mask_mat[y,x] = color[0:3] # 0:3 is to get "RGB" from "RGBA"
        ax.imshow(mask_mat)
        return mask_mat
    
    def _plot_masks_v2(self,ax,img,r_n, z,remaining_z, id_to_color,cmap,id_postfix,mask_mat=None):
        _, z_l = self._z_window(z,r_n,img=img)
        total_z = []
        total_z.append(z)
        total_z.extend(remaining_z)

        if mask_mat is None:
            mask_mat = np.zeros((z_l.shape[0],z_l.shape[1],3))
            mask_mat[:,:] = (1,1,1)
            for z in total_z:

                to_pop=set()
                _, z_l = self._z_window(z,r_n,img=img)
                for y in range(mask_mat.shape[0]):
                    for x in range(mask_mat.shape[1]):
                        c_id = str(int(z_l[y,x]))
                        c_id = f"{c_id}_{id_postfix}"
                        if c_id in id_to_color:
                            color = cmap(id_to_color[c_id])
                            mask_mat[y,x] = color[0:3] # 0:3 is to get "RGB" from "RGBA"
                            to_pop.add(c_id)
                for c_id in to_pop:
                    id_to_color.pop(c_id)
                    
        ax.imshow(mask_mat)
        return mask_mat
    
    def _create_group_colors(self, cmap,groups):
        n_colors = len(cmap.colors)
        group_colors = {}
        for i,group in enumerate(groups):
            group_colors[group] = i%n_colors
        return group_colors

    def _create_legend_elements(self,group_colors, cmap,markersize,group_name=None):
        legend_elements = []
        if group_name == "RGCs":
            group_colors={"Amacrine":14, "Rest are RGC Types":-1}
            cmap=plt.get_cmap('tab20')
        for group in group_colors:
            if group_colors[group] == -1:
                facecolor='white'
            else:
                facecolor = cmap(group_colors[group])
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=group,
                          markerfacecolor=facecolor, markeredgecolor='white',markersize=markersize))
        return legend_elements


    def visualize_classes(self,adata,group_col, z,r_n,remaining_z,id_postfix, cmap, img_cmap='Greys', show_legend=True, group_colors=None,group_name=None,keep_by_col=True, id_filter=None, plot_height=10,axis_fontsize=12,linewidth=1,markersize=13,frameon=True, stain="Cellbound2",plot_boundaries=True,bbox_to_anchor=(1.05,1.0),invert_yaxis=False,fname=None):
        mpl.rcParams['font.sans-serif'] = "Arial"
        mpl.rcParams['font.family'] = "sans-serif"
        img,z_l = self._z_window(z,r_n,stain=stain)
        n_cols=3
        fig,ax=plt.subplots(ncols=n_cols,nrows=1,figsize=(plot_height*4/3*n_cols, plot_height))
        ax[0].imshow(img,cmap=img_cmap)
        ax[1].imshow(img,cmap=img_cmap)
        blank_mat =  np.zeros((z_l.shape[0],z_l.shape[1],3))
        blank_mat[:,:] = (1,1,1)
        ax[2].imshow(blank_mat)

        if group_colors is None:
            groups = np.unique(adata.obs[group_colors])
            group_colors = self._create_group_to_color(cmap,groups)
        colors = adata.obs[group_col].map(group_colors).dropna()
        id_to_color = dict(zip(colors.index,colors.values.astype(int)))




        outlines,outlines_stitch_lst,masks_drawn=None,None,id_filter
        if plot_boundaries:
            outlines, outlines_stitch_lst,masks_drawn = self._plot_outlines(ax[1],z_l,r_n, remaining_z=remaining_z, masks_drawn=masks_drawn, outlines=outlines,outlines_stitch_lst=outlines_stitch_lst, color='red',stitch_color='blue',linewidth=linewidth,stain=stain)
            outlines, outlines_stitch_lst,_ = self._plot_outlines(ax[2],z_l,r_n, remaining_z=remaining_z, masks_drawn=masks_drawn, outlines=outlines,outlines_stitch_lst=outlines_stitch_lst, color='grey',stitch_color='grey',linewidth=linewidth,stain=stain)
        # self._plot_masks(ax[2],z_l,id_to_color,cmap,id_postfix=id_postfix)
        self._plot_masks_v2(ax[2],img, r_n,z,remaining_z,id_to_color,cmap,id_postfix=id_postfix)

        self._adjust_axes(img.shape, ax,r_n,axis_fontsize,adjust_bounds=True,invert_yaxis=invert_yaxis)
        legend_elements = self._create_legend_elements(group_colors=group_colors,cmap=cmap,markersize=markersize,group_name=group_name)
        if show_legend:
            ax[2].legend(handles=legend_elements,prop={'size': 18},bbox_to_anchor=bbox_to_anchor,frameon=frameon)

        if group_name is None:
            group_name="Default"

        if fname is None:          
            fname=os.path.join(self.figure_dir, f"TEST: {group_name}_showcase_z{z}_r{r_n}.png")
        else:
            fname=os.path.join(self.figure_dir, fname)

        plt.savefig(fname,bbox_inches='tight',dpi=200)
        print(fname)

    def calculate_density(self,adata_merfish,r_n,z=None):
        df=adata_merfish.obs
        XL_m,XU_m,YL_m,YU_m = self.micron_regions[r_n].reshape(-1)
        XL,XU,YL,YU = self.mosaic_regions[r_n].reshape(-1)
        if z is not None:
            z_l = self.stitched_m[z]
            z_l = z_l[YL:YU,XL:XU]
            cell_ids = [str(int(x)) for x in np.unique(z_l)]
            cell_ids.remove("0")
            cell_ids = set(cell_ids).intersection(set(df.index)) # Some cell_ids in the stitched matrix arent in the df since those cells have 0 transcripts and are excluded in the assignment part
        else:
            cell_ids = set(df.index)


        area_filt = (df['center_x'] >= XL_m) & (df['center_x'] <= XU_m) & (df['center_y'] >= YL_m) & (df['center_y'] <= YU_m)
        df_pc = df[df['pass_cutoff'] & (df['Size'] >= 50) & area_filt & df.index.isin(cell_ids)]
        n_pc = len(df_pc)

        n_a = len(df_pc[df_pc['is_amacrine'] & ~df_pc['is_RGC']])
        n_rc = len(df_pc[df_pc['is_RGC'] & ~df_pc['is_amacrine'] & (df_pc['cluster_names'] != 'Unassigned')])
        n_rnc = len(df_pc[df_pc['is_RGC'] & ~df_pc['is_amacrine'] & (df_pc['cluster_names'] == 'Unassigned')])
        n_b = len(df_pc[df_pc['is_RGC'] & df_pc['is_amacrine']])
        n_n = len(df_pc[~df_pc['is_RGC'] & ~df_pc['is_amacrine']])

        area_mm2 = (XU_m-XL_m)*(YU_m-YL_m)/(10**6) # Convert from micro^2 to milli^2
        print(f"Number of cells in {np.abs(YL_m-YU_m)} by {np.abs(XL_m-XU_m)} area: {len(df_pc)}")
        print("Overall cell density:", n_pc/area_mm2)
        print("RGC Density:", n_rc/area_mm2, "Cells:", n_rc) # Units: mm^2
        print("Amacrine Density:", n_a/area_mm2, "Cells:", n_a)
        
        """
        print("Number of cells with good amount of transcripts and not too small: ", n_pc)
        print("")
        print(f"# cells that pass amacrine marker cutoff only: {n_a} ({n_a/n_pc})")
        print(f"# RGCs classified: {n_rc} ({n_rc/n_pc})")
        print(f"# RGCs not classified: {n_rnc} ({n_rnc/n_pc})")
        print(f"# cells that pass both RGC and amacrine cutoffs: {n_b}, ({n_b/n_pc})")
        print(f"# of non-RGC/non-amacrine: {n_n} ({n_n/n_pc})")
        """
    

        
