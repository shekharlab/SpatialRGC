import os
import importlib
import re 
import pickle


import cv2
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from shapely.geometry import Polygon,Point
from matplotlib.lines import Line2D
from collections import OrderedDict
import matplotlib as mpl
import xgboost as xgb

import spatial_rgc.utils.classifier as cl
import spatial_rgc.utils.viz_tools as vz
import spatial_rgc.utils.preprocess_utils as pr
import spatial_rgc.utils.assorted_utils as utils

import spatial_rgc.utils.model_constants as constants




# Functions outside class might be useful in other contexts

class AggregatePipeline:
    #TODO: Consider moving many of these constants to their own file
    
    DISTRO_FIGURES = "distro_analysis"
    TRAIN_DICT = {'pc/Am+/RGC+': 0, 'pc/Am+/RGC-':1, 'pc/Am-/RGC+':2, 'pc/Am-/RGC-':3,'dpc/Am+/RGC+': 4, 'dpc/Am+/RGC-':5, 'dpc/Am-/RGC+':6, 'dpc/Am-/RGC-':7}
    RGC_MARKERS= ["Rbpms", "Pou4f1", "Pou4f2", "Slc17a6"]
    AMACRINE_MARKERS = ["Tfap2a", "Tfap2b", "Chat"]
    ENDO_MARKERS = ['Tagln2','Klf2','Anxa3']
    ASTRO_MARKERS = ['S100b','Igfbp5','Sox2']
    KEEP_COL= "to_keep"
    CUTOFF_COL="pass_cutoff"
    GROUP_MAP = {'pc/Am+/RGC+': "Other", 'pc/Am+/RGC-':"Amacrine", 'pc/Am-/RGC+':"RGC", 'pc/Am-/RGC-':"Other",'dpc/Am+/RGC+': "N/A", 'dpc/Am+/RGC-':"N/A", 'dpc/Am-/RGC+':"N/A", 'dpc/Am-/RGC-':"N/A"}
    GROUP_MAP_CLUBBED = {'pc/Am+/RGC+': "Amacrine", 'pc/Am+/RGC-':"Amacrine", 'pc/Am-/RGC+':"RGC", 'pc/Am-/RGC-':"Amacrine",'dpc/Am+/RGC+': "N/A", 'dpc/Am+/RGC-':"N/A", 'dpc/Am-/RGC+':"N/A", 'dpc/Am-/RGC-':"N/A"}# Group "Other" into "Amacrine", KS request

    CMAP = plt.get_cmap("Paired")
    CLASS_COLOR_MAP = {"RGC":CMAP(3),"Amacrine":CMAP(7), "Other":CMAP(9)}
    NUM_RGC_TYPES = 45
    MEDIUM_FIG_SIZE = (6.4*2,4.8*2)
    EXCEPTIONS={"control"}



    def __init__(self,subdirectories,removal_areas_per_region=None, anndata_f="Cellpose_model_c2knl_full_retina.h5ad",base_dir=None,figure_fold="figures",
                data_fold="data",aggregate_fold='intermediate_figures',feature_names_f="models/150g/feature_names.pickle", xgb_model_f="models/150g/xgb_150g",rgc_median_transcripts=176):
        prev_run = None
        self.processed_regions=dict() # keys = region_#, values = preprocessed region data
        for subd in subdirectories:
            n_genes,run,rg = self._check_valid_subd(subd)
            self.processed_regions[rg] = None
            # assert (run == prev_run) or prev_run==None, print(run,prev_run)
            prev_run = run
        self.run_fold = f"{n_genes}g_rn{run}"
        self.subdirectories = subdirectories
        self.removal_areas_per_region = removal_areas_per_region
        self.anndata_f = anndata_f
        if base_dir is None:
            self.base_dir=constants.BASE_DIR
        else:
            self.base_dir=base_dir
        
        self.figure_dir= os.path.join(self.base_dir,figure_fold)
        self.aggregate_dir = os.path.join(self.base_dir,aggregate_fold)
        self.data_dir = os.path.join(self.base_dir, data_fold)
        self.merged_ann=None
        self.filt= False
        self.feature_names_f = feature_names_f
        self.xgb_model_f=xgb_model_f
        self.rgc_median_transcripts = rgc_median_transcripts

    def merge_region_dfs(self,alignment,stain,im_dir,keys=None, merged_f='Cellpose_merged.h5ad',threshold_am=3,threshold_rgc=10,min_transcripts=15) -> None:
        """  Concatenates each separate region dataframes, updates corresponding class parameter, and saves to disk
        Args:
            merged_f (str): File name for merged anndata object
            alignment (dict): Keys are subdirectories, values are "center": (y,x),"z":z_layer,"downsample_ratio":dr,"rotation":rot} where y,x are in downsampled image coords after using ratio of dr
            stain (str): Staining used to align (Not really a required argument)
            im_dir (str): Directory of images used to align (corresponds to specific z_layer)

        Returns:
            None
        Updates:
            self.merged_ann (anndata): Class parameter storing the concatenated anndata and aligned center coordinates
        Writes:
            Merged anndata object with all regions data concatenated and center coords aligned
        """

        ann_arr = []
        for i,subd in enumerate(self.subdirectories):
            adata= sc.read_h5ad(os.path.join(self.data_dir, subd, self.anndata_f))
            with open(self.feature_names_f,"rb") as f:
                feature_names = pickle.load(f)
            adata=adata[:,feature_names].copy()
            if alignment is not None:
                self._align_coordinates(adata, alignment,stain,im_dir, subd,coord_prefix='aligned',idx=i)
            xy_arr = adata.obs[['center_x','center_y']].values
            
            inside_arr = np.ones(xy_arr.shape[0])
            removal_areas=self.removal_areas_per_region[subd]
            if removal_areas is not None:
                inside_arr = self._check_pts(xy_arr,removal_areas)
            adata.obs[self.KEEP_COL] = inside_arr
            adata.obs[self.KEEP_COL] = adata.obs[self.KEEP_COL].astype(bool)

            ann_arr.append(adata)

        if keys is None:
            keys=range(len(self.subdirectories))
        merged_ann = ad.concat(ann_arr,axis=0,label='region', index_unique='_rg', keys =keys)
        merged_ann.layers['raw'] = merged_ann.X.copy()
        self.merged_ann=merged_ann
        self.merged_ann.layers
        self._annotate_anchors(threshold_am=threshold_am, threshold_rgc=threshold_rgc,min_transcripts=min_transcripts)

        output_dir = os.path.join(self.data_dir,self.run_fold)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        if merged_f is not None:
            print(self.merged_ann.obs['is_amacrine'])
            self.merged_ann.write_h5ad(os.path.join(output_dir,merged_f))
   
    def _check_valid_subd(self, subd):
        """ Checks that subdirectory string is in appropriate format and returns the info encoded in it
        Args:
            subd (str): Subdirectory in format of f'{#_genes}g_rn{run_#}_rg{region_#}'
        Return:
            #_genes,run_#,region_# (str,str,str)
        """
        for e in self.EXCEPTIONS:
            if re.match(e,subd) is not None:
                return -1,-1,-1
        match = re.search(r"(.*)g_rn(.*)_rg(.*)",subd)
        assert match is not None, 'Not valid string format'
        for i in range(1,4):
            assert match.group(i).isnumeric()
        
        return match.group(1), match.group(2), match.group(3)

    def _check_pts(self, xy_arr, polygon_lst):     
        """ Finds which (x,y) are contained within at least one polygon denoted by a list of vertex
        Args:
            xy_arr (np.array):  Array of (x,y) coordinates with shape (n,2)
            polygon_lst (list of lists containing tuples): Each sublist contains a set of coordinates which form the vertices of a polygon
        
        Return:
            keep_arr (np.array): An (n,) array, where 1 represents that corresopnding point in xy_arr is contained in at least one of the polygons in polygon_lst
        """
        keep_arr = np.ones(xy_arr.shape[0])
        for i,pt in enumerate(xy_arr):
            for poly in polygon_lst:
                if Polygon(poly).contains(Point(pt)):
                    keep_arr[i] = 0
                    break
        return keep_arr

   
    def _preprocess(self, adata,target_sum=None):
        """Takes anndata and updates it IN-PLACE following a classic preprocessing 
        NOTE: Currently does not support batch correction; consider adding argument for that; 
        NOTE: Consider updating to support in-place and not in-place as well
        Args:
            adata (anndata): Anndata with parameter 'X' containing raw counts
            target_sum (int): Number of transcripts to normalize to (176 is the value of the 139 genes in the RGC dataset)
        Returns:
            None
        Updates:
            adata (anndata) by adding raw counts as a layer, and updating parameter 'X' in-place
        """
        adata.layers['raw'] = adata.X.copy()
        sc.pp.normalize_total(adata,target_sum=target_sum, inplace=True)
        print(np.min(adata.X))
        sc.pp.log1p(adata, copy=False)
        sc.pp.scale(adata,copy=False)

    def _annotate_anchors(self,adata=None, threshold_am=3,threshold_rgc=10,min_transcripts=15, min_size=50, removal_areas=None):
        """Takes anndata and updates it IN-PLACE by adding columns denoting which cells pass high marker cutoffs (and thus act as anchors)
        Args:
            adata (anndata): Anndata object to filter; alternatively if None just tries to use merged_anndata instead; Expects 'X' to be raw counts
            threshold_am (int): Marker cutoff for amacrine cell
            threshold_rgc (int): Marker cutoff for RGC
            min_transcripts (int): Minimum number of transcripts required to pass cutoff
            min_size (int): Minimum size to pass cutoff
        NOTE: Consider updating to support in-place and not in-place as well
        Args:
            adata (anndata): Anndata with parameter 'X' containing raw counts
            target_sum (int): Number of transcripts to normalize to (176 is the value of the 139 genes in the RGC dataset)
        Returns:
            None
        Updates:
            adata (anndata) by 1)Adding raw counts as a layer 2) Updating parameter 'X' in-place, 3) Adding cutoff columns, 4) Adding 'group' and 'group num' columns
        """
        if adata is None:
            adata=self.merged_ann

        gene_counts = adata.X.sum(axis=1)
        is_am = np.array(np.sum(adata[:,self.AMACRINE_MARKERS].X, axis=1) >= threshold_am)
        adata.obs['is_amacrine'] = is_am.astype(bool)
        is_rgc = np.array(np.sum(adata[:,self.RGC_MARKERS].X,axis=1) > threshold_rgc)
        adata.obs['is_RGC']=is_rgc.astype(bool)
        pass_cutoff = (gene_counts >= max(min_transcripts, np.percentile(gene_counts, 10))) & (adata.obs['Size'] >= min_size)


        adata.obs[self.CUTOFF_COL] = pass_cutoff
        adata.obs['counts'] = gene_counts
        adata.obs['group'] = 'None'

        adata.obs.loc[is_am & is_rgc & pass_cutoff,'group'] = 'pc/Am+/RGC+'
        adata.obs.loc[is_am & ~is_rgc & pass_cutoff,'group'] = 'pc/Am+/RGC-'
        adata.obs.loc[~is_am & is_rgc & pass_cutoff,'group'] = 'pc/Am-/RGC+'
        adata.obs.loc[~is_am & ~is_rgc & pass_cutoff,'group'] = 'pc/Am-/RGC-'

        adata.obs.loc[is_am & is_rgc & ~pass_cutoff,'group'] = 'dpc/Am+/RGC+'
        adata.obs.loc[is_am & ~is_rgc & ~pass_cutoff,'group'] = 'dpc/Am+/RGC-'
        adata.obs.loc[~is_am & is_rgc & ~pass_cutoff,'group'] = 'dpc/Am-/RGC+'
        adata.obs.loc[~is_am & ~is_rgc & ~pass_cutoff,'group'] = 'dpc/Am-/RGC-'

        adata.obs['group'] = pd.Categorical(adata.obs['group'],categories=self.TRAIN_DICT.keys())


        adata.obs['group_num'] = adata.obs['group'].map(self.TRAIN_DICT)
    
       
    def _remap_cell_groups(self,adata_f,cluster_col='leiden', group_col='group_num', new_col='mapped_group',rgc_upper_cut=0.70,rgc_lower_cut=0.1):
        """Takes anndata, mapping of cell types/classses to integers, and uses clustering labels to remap cell types/classes based off cluster proportions
        NOTE: Consider updating to support in-place and not in-place as well
        Args:
            adata (anndata): Anndata with parameter 'X' containing preprocessed data
            cluster_col (str): Denotes column to look at clustering for
            group_col (str): Column name of numeric labeling of original types/classes
            new_col (str): Column name to add with updated types/classes
        Updates
            adata (annadata): Anndata with keys added for a new column called 'mapped_group', which contains the new assignments
            merged_ann (anndata): Updates 'group' key in obs with 'mapped_group'

        """
        adata_f.obs[group_col]
        if new_col in adata_f.obs.columns:
                adata_f.obs.drop(columns=new_col,inplace=True)
        adata_f.obs[new_col] = adata_f.obs[group_col].copy()

        for clust_id in np.unique(adata_f.obs[cluster_col]):
            adata_fc = adata_f[adata_f.obs[cluster_col] == clust_id]
            minlength = len(self.TRAIN_DICT)
            pct_distro = np.bincount(adata_fc.obs[group_col], minlength=minlength)/len(adata_fc)
            double_pct, am_pct,rgc_pct, n_pct,_,_,_,_ = pct_distro
            if rgc_pct >= rgc_upper_cut:
                to_update = [0,1,3]
                adata_f.obs.loc[(adata_f.obs[cluster_col] == clust_id) & (adata_f.obs[group_col].isin(to_update)),new_col] = 2 # Update to all be RGC
            elif rgc_pct < rgc_lower_cut and (n_pct<=0.5):
                to_update=[0,3]
                adata_f.obs.loc[(adata_f.obs[cluster_col] == clust_id) & (adata_f.obs[group_col].isin(to_update)),new_col] = 1 # Update all non-RGC to be amacrine
        inv_dict= {v:k for k,v in self.TRAIN_DICT.items()}
        adata_f.obs[new_col] = pd.Categorical(adata_f.obs[new_col].map(inv_dict),categories=self.TRAIN_DICT.keys())
        self.merged_ann.obs['group'].update(adata_f.obs[new_col])


    # Pipeline to create
    def create_group_map(self, filt, subd, is_region, colors=['group','mapped_group'], n_neighbors=30,steps=2,output_f="all_cells_reassigned.h5ad"):
        """Takes anndata, preprocesses, clusters and creates umap on it
        NOTE: Consider updating to support in-place and not in-place as well
        Args:
            adata (anndata): Anndata with anchor types in observation columns (From _annotate_anchors); If none-passed, tries to use parameter merged_ann
            color ([str, ]): Columns in adata.obs.columns to color umap plots with
            n_neighbors (int): # of neighbors for neighborhood graph construction
            steps (int): Run only specific parts of program
            ouptut_f (str): Filename of output anndata (Inputting None means nothing will be saved)
        Returns
            adata (annadata): Anndata with keys added for neighbors,umap, leiden clustering, and remapped cell classes
        Writes:
            UMAP with colorcoding based off keys in 'colors' parameter
            Final anndata object after processing (if output_f is not None)

        """
        adata_f = self.merged_ann[filt].copy()
        # print(adata_f.obs)
        if steps >=1:
            self._preprocess(adata_f,target_sum=self.rgc_median_transcripts)
            sc.pp.neighbors(adata_f, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.umap(adata_f)
        
        if steps>=2:
            sc.tl.leiden(adata_f, resolution=1,key_added='leiden')
            self._remap_cell_groups(adata_f)
            with plt.rc_context():
                sc.pl.umap(adata_f, color=colors,show=False)
                if is_region:
                    output_dir = os.path.join(self.figure_dir,subd,self.DISTRO_FIGURES)
                else:
                    output_dir = os.path.join(self.aggregate_dir,subd)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(os.path.join(output_dir,"mapped_vs_unmapped.png"), bbox_inches="tight")

            test_dict = {f"M{i}":i for i in np.unique(adata_f.obs['leiden'].values.astype(int))}
            utils.plot_mapping_new(test_labels=adata_f.obs['leiden'].values.astype(int), 
                            test_predlabels=adata_f.obs['group_num'].values.astype(int),
                            test_dict=test_dict, 
                            train_dict=self.TRAIN_DICT,
                            re_order=False,
                            re_index=True,
                            xaxislabel='Type (strong markers)', yaxislabel='MERFISH Leiden Cluster',
                            save_as=os.path.join(output_dir,"Clustering vs. type confusion matrix.png"))
                            
            if output_f is not None:
                adata_f.write(filename=os.path.join(self.data_dir,subd, output_f))
        return adata_f
        
    
    def create_final_assignments(self,type_col='cluster',class_col='class', new_col='final_assignment'):
        inv_dict = {v:k for k, v in self.train_dict.items()}
        types = self.adata_cache.obs[type_col].map(inv_dict)

        self.merged_ann.obs[new_col] = self.merged_ann.obs[class_col].copy()
        self.merged_ann.obs[new_col] = self.merged_ann.obs[new_col].cat.add_categories(self.train_dict.keys())

        self.merged_ann.obs[new_col].update(types)
        self.merged_ann.obs[new_col] = self.merged_ann.obs[new_col].replace(to_replace={"RGC":"Removed RGC"})
        print(self.merged_ann.obs)

    
    
    def _align_coordinates(self,adata, alignment,stain,im_dir,subdirectory,coord_prefix='aligned',idx=None):
        """ Takes anndata and in IN-PLACE fashion adds columns denoting transformed coordiante system with respect to base_idx=0
        Args:
            adata (anndata): Anndata with obs keys of ['center_x','center_y]
            alignment (dict): See merged_region_dfs for more details
            stain (str): See merged_region_dfs for more details
            im_dir (str): See merged_region_dfs for more details

        Updates:
            adata (anndata): Adds new columns (names are prepended with coord_prefix_)

        """
        alignment = OrderedDict(sorted(alignment.items()))
        a= vz.create_aligner(alignment,stain,im_dir)
        base_idx= 0
        dr = alignment[subdirectory]["downsample_ratio"]
        _,_,rg = self._check_valid_subd(subdirectory)

        if idx is None:
            idx=int(rg)

        rg = int(rg)

        t = a.get_align_transform(base_idx,idx,dr=dr,mappings_path=os.path.join(self.base_dir,"imaging_scripts","mappings"))
        rot_degree=(360-a.rotations[idx]) # a.rotations is in pixels, convert to respect to micron orientation
        M = vz.rotation_matrix(rot_degree)
        transf = adata.obs[['center_x','center_y']].values+[t[0], t[1]] - a.centers_micron[base_idx]
        transf = (M@ transf.T).T + a.centers_micron[base_idx]
        new_cols=[f"{coord_prefix}_{col}" for col in ['center_x','center_y']]
        adata.obs[new_cols] = transf  

    def plot_scatter(self,filt,invert_yaxis=False,invert_xaxis=False):
        """Uses filter to plot spatial distribution of specific cells (and their classes)
        Args:
            filter (pd.Series): Boolean series used to index paramter merged_ann
        """
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=self.MEDIUM_FIG_SIZE)
        df= self.merged_ann[filt].obs
        colors = df['group'].map(self.GROUP_MAP).map(self.CLASS_COLOR_MAP).values
        ax.scatter(df['aligned_center_x'],df['aligned_center_y'],s=0.5,c=colors)
        legend_elements=[]
        for group in self.CLASS_COLOR_MAP:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=group,
                                markerfacecolor=self.CLASS_COLOR_MAP[group], markersize=13))
        ax.legend(handles=legend_elements)
        ax.set_xlabel(r'x ($\mu m$)')
        ax.set_ylabel(r'y ($\mu m$)')
        if invert_yaxis:
            ax.invert_yaxis()
        if invert_xaxis:
            ax.invert_xaxis()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.savefig(os.path.join(os.path.join(self.aggregate_dir,self.run_fold,"Class_spatial_scatter.png")))    
        plt.show()
    
    def plot_umap(self,adata_f):
        """Plots UMAP of cells in adata_f, as well as corresponding markers for cell classes
        Args:
            adata_f (anndata): Anndata already filtered for desired cells, with key 'X_umap' present in obsm already.
        """
        
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(6.4*2,4.8*2))
        colors = adata_f.obs['mapped_group'].map(self.GROUP_MAP).map(self.CLASS_COLOR_MAP).values
        ax.scatter(adata_f.obsm['X_umap'][:,0],adata_f.obsm['X_umap'][:,1],s=0.5,c=colors)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        legend_elements=[]
        for group in self.CLASS_COLOR_MAP:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=group,
                                markerfacecolor=self.CLASS_COLOR_MAP[group], markersize=13))
        ax.legend(handles=legend_elements)
        plt.savefig(os.path.join(os.path.join(self.aggregate_dir,self.run_fold,"Class_umap.png")))
        plt.show()
        with plt.rc_context():
            R_sum_col = 'Total RGC Marker Count (Rbpms/Slc17a6/Pou4f1/Pou4f2)'
            A_sum_col = 'Total Amacrine Marker Count (Tfap2a/Tfap2b/Chat)'
            adata_f.obs[R_sum_col] = np.sum(adata_f[:,self.RGC_MARKERS].layers['raw'],axis=1)
            adata_f.obs[A_sum_col] = np.sum(adata_f[:,self.AMACRINE_MARKERS].layers['raw'],axis=1)
            # sc.pl.umap(adata_f,color=self.RGC_Markers+self.AMACRINE_MARKERS,layer='raw',vmin=0,vmax=30,show=False)
            sc.pl.umap(adata_f,color=[R_sum_col,A_sum_col],vmin=0,vmax=20,show=False)
            plt.savefig(os.path.join(os.path.join(self.aggregate_dir,self.run_fold,"marker_sum_umap.png")))
        with plt.rc_context():
            sc.pl.umap(adata_f,color=self.RGC_MARKERS+self.AMACRINE_MARKERS,layer='raw',vmin=0,vmax=10,show=False)
            plt.savefig(os.path.join(os.path.join(self.aggregate_dir,self.run_fold,"marker_umap.png")))
    
    def plot_class_counts(self,filt):
        """Uses filter to select certain cells to calculate cell class distribution
        Args:
            filter (pd.Series): Boolean series used to index paramter merged_ann
        """
        mpl.rcParams['font.sans-serif'] = "Arial"
        mpl.rcParams['font.family'] = "sans-serif"
        df=self.merged_ann[filt].obs
        type_counts=np.zeros(3)
        RGC_idx,Amacrine_idx,Other_idx =0,1,2
        type_counts[RGC_idx] = np.sum(df['group'].map(self.GROUP_MAP)=="RGC")
        type_counts[Amacrine_idx] = np.sum(df['group'].map(self.GROUP_MAP)=="Amacrine")
        type_counts[Other_idx] = np.sum(df['group'].map(self.GROUP_MAP)=="Other")

        df['group'].map(self.GROUP_MAP).value_counts().plot(kind='bar',color=[self.CMAP(2),self.CMAP(6),self.CMAP(8)],width=0.75)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel("Number of cells",fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()

        plt.savefig(os.path.join(os.path.join(self.aggregate_dir,self.run_fold,"cell_class.png")),bbox_inches='tight')
    
    def _rearrange_and_preprocess(self,adata,feature_names_f,target_sum=None):
        """Rearranges var.index order to match XGBoost model training set, and transforms data
        Args:
            adata_f (anndata): Anndata with key 'raw' for layers
            filt (str): Corresponds to group of cells in values of self.GROUP_MAP
        Returns:
            adata_f (anndata): Modified anndata 
        """
        with open(feature_names_f,"rb") as f:
            feature_names = pickle.load(f)
        adata= adata[:,feature_names].copy()
        adata.X = adata.layers['raw'].copy()
        self._preprocess(adata,target_sum=target_sum)
        return adata
    
    def _cluster(self,adata,n_neighbors=30,resolution=0.8,key_added='leiden',override=False):
        """Clusters in-place
        Args:
            adata_f (anndata): Should already be preprocessed
        Modifies:
            Updates adata_f in-place with neighborhood graph and clustering
        """
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X',copy=False)
        sc.tl.umap(adata,copy=False)
        sc.tl.leiden(adata, resolution=resolution,key_added=key_added,copy=False)
    
    def _classify(self,adata, class_col,num_classes,unassigned_indicator,xgb_model_f,cutoff=0.5,train_dict=None):
        """Classifies (RGC) types in place
        Args:
            adata_f (anndata): Should already be preprocessed
            class_col (str): Name of column containing type prediction
            num_classes (int): Number of unique classes XGBoost saw during training
            unassigned_indicator (int): Number indicating what number should be assigned to cells where no class is above a certain probability threshold (0.5)
        Modifies:
            Updates adata_f in-place with new column and 'y_probs' key in obsm containing probability scores for predictions
        """
        if train_dict is None:
            train_dict={f"C{i+1}":i for i in range(num_classes)}
            train_dict['Unassigned'] = unassigned_indicator
            assert unassigned_indicator not in range(num_classes), "Unassigned indicator # is the same as one of the class #'s"

        model = xgb.Booster()
        model.load_model(xgb_model_f)

        y_pred,y_probs = utils.cutoff_predict(X=adata.X, model=model, unassigned_indicator=unassigned_indicator, cutoff=cutoff)
        adata.obsm['y_probs'] = y_probs
        adata.obs[class_col] = y_pred
        return train_dict
    
     #TODO: Have RGC classifications take anndata as argument optionally and move to helper class
    def generate_RGC_classifications(self,filt=None,cutoff=0.5):
        if filt is None:
            df = self.merged_ann.obs
            # filt = (df['group'].map(self.GROUP_MAP)=="RGC") & df['to_keep'] & df['pass_cutoff']
            filt = (df['class']=="RGC") & (df['to_keep']) & (df['pass_cutoff'])

        adata = self.merged_ann[filt]
        adata = self._rearrange_and_preprocess(adata,feature_names_f=self.feature_names_f,target_sum=self.rgc_median_transcripts)
        self._cluster(adata)
        train_dict = self._classify(adata,class_col='cluster',num_classes=self.NUM_RGC_TYPES,unassigned_indicator=self.NUM_RGC_TYPES,xgb_model_f=self.xgb_model_f,cutoff=cutoff,train_dict=self.train_dict)
        #self.class_train_dict = train_dict
        self.adata_cache=adata
    
    def plot_confusion_matrix(self,train_key='cluster', test_key='leiden',re_order_cols=None,save_f=None):
        assert self.adata_cache is not None, "Generate RGC classifications first"
        num_clusters = len(np.unique(self.adata_cache.obs[test_key]))
        self.class_test_dict = {f"M{i}":int(i) for i in range(num_clusters)}
        re_order=False
        if re_order_cols is not None:
            re_order=True

        if save_f is None:
            save_f= os.path.join(self.aggregate_dir,self.run_fold, "confusion_matrix.png")
        with plt.rc_context({"font.size":14, "figure.figsize":[20,20],"figure.dpi":300}):
            utils.plot_mapping_new(test_labels=self.adata_cache.obs[test_key].values.astype(int), #y-axis
                            test_predlabels=self.adata_cache.obs[train_key].values,#x-AXIS
                            test_dict=self.class_test_dict,  # y-axis
                            train_dict=self.class_train_dict, # x-axis
                            re_order=re_order,
                            re_order_cols=re_order_cols,
                            re_index=True,
                            xaxislabel='scRNA-seq', yaxislabel='MERFISH Leiden Cluster',
                            save_as=save_f
            )
    
    def plot_classification_threshold(self,title="",fname="Model_probability_comparisons.png"):
        adata= self.adata_cache
        # Compare cell type size distributions
        fontsize=24
        y_probs = adata.obsm['y_probs']
        ordering = [t[0] for t in (sorted(self.class_train_dict.items(),key=lambda t: t[1]))]# Create ordered list of keys in train_dict sorted using values of train_dict

        adata.obs['cluster_probs'] = np.max(y_probs, axis=1)
        adata.obs['cluster_no_threshold'] = (np.argmax(y_probs,axis=1) + 1).astype(str)
        adata.obs['cluster_no_threshold'] = adata.obs['cluster_no_threshold'].apply(lambda x: 'C' + x)
        adata.obs['cluster_no_threshold'] = pd.Categorical(adata.obs['cluster_no_threshold'],categories=ordering)
        adata.obs.boxplot(column="cluster_probs",by="cluster_no_threshold",figsize=(20,6),showfliers=False,flierprops={'markersize':2})
        plt.plot(np.arange(46) ,[0.5]*46,color='r',label='Model threshold',linestyle='dashed',alpha=0.7)
        plt.plot(np.arange(46) ,[0.02]*46,color='grey',label='Random assignment',linestyle='dashed',alpha=0.7)

        plt.suptitle('')
        plt.ylabel(r'Maximum probability assignment',fontsize=fontsize)
        plt.xlabel(r'RGC Type',fontsize=fontsize)
        plt.title('')
        plt.grid(visible=None)
        plt.tick_params(axis='x',labelsize=fontsize,rotation=90)
        plt.tick_params(axis='y',labelsize=fontsize)
        # plt.yticks(ticks=np.arange(0,1.1,0.1))
        plt.xlim(0,46)
        plt.minorticks_off()
        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('left')
        plt.legend(loc='best')
        plt.title(title)
        plt.savefig(os.path.join(self.aggregate_dir, self.run_fold,fname),dpi=200,facecolor='w',bbox_inches='tight')
        plt.show()
    
    def plot_spatial_group(self,adata, values, group_names, other_names=None, col='cluster',title="", fname="spatial_cluster_subset.png",s_group=10,s_other=0.1,colors='Red',invert_yaxis=False,invert_xaxis=True): # default is for first region
        # Compare cell type size distributions
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=self.MEDIUM_FIG_SIZE)
        df = adata[adata.obs[col].isin(values)].obs
        """
        if type(colors[0]) == tuple: # Check if color channel
            n_channels = len(colors[0])
            color_arr = np.zeros((df.shape[0],n_channels))
            m=dict(zip(values,colors))
            for i in range(n_channels):
                color_arr[:,i] = df[col].apply(lambda x: m[x][i]).values
            colors=color_arr
        """
        for i,value in enumerate(values):
            df_pl = df[df[col]==value]
            ax.scatter(df_pl['aligned_center_x'],df_pl['aligned_center_y'],s=s_group,color=colors[i],label=group_names[i])


        if other_names is not None:
            df = adata[~adata.obs[col].isin(values)].obs
            ax.scatter(df['aligned_center_x'],df['aligned_center_y'],s=s_other,color='Grey',label=other_names)
        

        ax.set_facecolor('black')
        ax.legend()
        ax.set_axis_off()
        if invert_xaxis:
            ax.invert_xaxis()
        if invert_yaxis:
            ax.invert_yaxis()
        fig.savefig(os.path.join(self.aggregate_dir, self.run_fold,fname),dpi=200,facecolor='white',bbox_inches='tight')
        fig.show()
    
    def plot_all_umap(self, cluster_key='leiden',group_key='group_num',fname="all_cells_reassigned.h5ad"):
        adata= sc.read_h5ad(os.path.join(self.data_dir,self.run_fold,fname))
        print(cluster_key,group_key)
        self.class_test_dict = {f"M{i}":int(i) for i in np.unique(adata.obs[cluster_key])}
        utils.plot_mapping_new(test_labels=adata.obs[cluster_key].values.astype(int), 
                            test_predlabels=adata.obs[group_key].values.astype(int),
                            test_dict=self.class_test_dict, 
                            train_dict=self.TRAIN_DICT,
                            re_order=False,
                            re_index=True,
                            xaxislabel='Type (strong markers)', yaxislabel='MERFISH Leiden Cluster',
                            save_as="")
        with plt.rc_context({"figure.dpi":(200)}):
            sc.pl.umap(adata,color=[group_key,'mapped_group'],show=False)
            plt.savefig(os.path.join(self.aggregate_dir,self.run_fold, "remapping.png"),bbox_inches='tight')
            plt.show()
        with plt.rc_context({"figure.dpi":(100)}):
            sc.pl.umap(adata,color=[cluster_key],show=False)
            plt.savefig(os.path.join(self.aggregate_dir,self.run_fold, "leiden_clusters.png"),bbox_inches='tight')
            plt.show()
        with plt.rc_context({"figure.dpi":(100)}):
            sc.pl.umap(adata,color=['region'],show=False)
            plt.savefig(os.path.join(self.aggregate_dir,self.run_fold, "regions_umap.png"),bbox_inches='tight')
            plt.show()

        adata.obs['mapped_clubbed_group'] = adata.obs['mapped_group'].map(self.GROUP_MAP_CLUBBED)
        cmap = plt.get_cmap('tab10')
        with plt.rc_context({"figure.dpi":(300)}):
            sc.pl.umap(adata,color='mapped_clubbed_group',palette={"RGC":cmap(1), "Amacrine":cmap(0)},show=False)
            plt.savefig(os.path.join(self.aggregate_dir,self.run_fold, "clubbed_classification.png"),bbox_inches='tight')
            plt.show()

    
    def get_filtered_statistics(self,s=0.01, markerscale=10,bbox_to_anchor=[1.0,1.0]):
        plt.figure(figsize=self.MEDIUM_FIG_SIZE)
        adata = sc.read_h5ad(os.path.join(self.data_dir,self.run_fold,"Cellpose_merged_remapped.h5ad"))
        print(adata.obs)
        filt = (adata.obs['to_keep']) & (~adata.obs["pass_cutoff"])
        filt2 = (adata.obs['to_keep']) & (adata.obs["pass_cutoff"])
        adata_f = adata[filt]
        adata_f2 = adata[filt2]
        plt.scatter(adata_f2.obs['aligned_center_x'],adata_f2.obs['aligned_center_y'],s=s,label='Kept cells')
        plt.scatter(adata_f.obs['aligned_center_x'],adata_f.obs['aligned_center_y'],s=s,label='Filtered cells (Low transcript/tiny)')
        plt.legend(markerscale=15)
        print(adata_f.obs['group'].value_counts())
        print(adata_f2.obs['group'].value_counts())
        plt.gca().invert_yaxis()
        plt.legend(bbox_to_anchor=bbox_to_anchor,markerscale=markerscale)
        plt.savefig(os.path.join(self.aggregate_dir,self.run_fold,"FIltered_unfiltered.png"),bbox_inches='tight')
        plt.show()

        
    
    
    




        

    
  

        
    
    
    