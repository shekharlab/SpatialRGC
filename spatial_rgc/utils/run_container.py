import os
import shutil
import pickle as pickle
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patches as patches
import scanpy as sc
import seaborn as sns
import anndata as ad
import scipy
import re


import spatial_rgc.utils.aggregate_pipeline as ap
import spatial_rgc.utils.rconstants as rc
import spatial_rgc.utils.plot_utils as pu
import spatial_rgc.utils.preprocess_utils as pr
import spatial_rgc.utils.visualizer as visualizer
import spatial_rgc.utils.classifier as cl
import spatial_rgc.utils.model_constants as constants



class RunContainer():
    RGC_ATLAS_F = os.path.join(constants.BASE_DIR,"data", "atlases", "RGCatlas.h5ad")
    #Below needs to be updated after every run
    ALIGNMENT_MAP = {
        "140g_rn3":rc.DAPI_ALIGNMENTS[0], "140g_rn4":rc.CELLBOUND2_ALIGNMENTS[0],"140g_rn5":rc.DAPI_ALIGNMENTS[1],
        "140g_rn6":rc.DAPI_ALIGNMENTS[2],"140g_rn8":rc.DAPI_ALIGNMENTS[3],"140g_rn9":rc.DAPI_ALIGNMENTS[4], "140g_rn7":rc.DAPI_ALIGNMENTS[5],"140g_rn10":rc.DAPI_ALIGNMENTS[6],"140g_rn12":rc.DAPI_ALIGNMENTS[8],"140g_rn14":rc.DAPI_ALIGNMENTS[9],"140g_rn15":rc.DAPI_ALIGNMENTS[10],
        "140g_rn16":rc.DAPI_ALIGNMENTS[11],"140g_rn17":rc.DAPI_ALIGNMENTS[12],"140g_rn18":rc.DAPI_ALIGNMENTS[13],"140g_rn23":rc.DAPI_ALIGNMENTS[14],"140g_rn25":rc.DAPI_ALIGNMENTS[15],"140g_rn26":rc.DAPI_ALIGNMENTS[16]
    }
    STAIN_MAP = {"140g_rn3":"DAPI", "140g_rn4":"Cellbound2","140g_rn5":"DAPI"}
    ATLAS_RUN="atlas"
    RENAMED_GENE_MAP = {'Fam19a4':'Tafa4','Tusc5':'Trarg1'} # keys are atlas gene names, values are merfish gene names
    TRAN_FIG_MAPPING = {'S2_S4': {'cluster_names':["C12","C16","C10","C24"],'genes':["Cartpt","Mmp17","Col25a1","Gpr88","Tafa4"]},
           'ipRGCs': {'cluster_names':["C33", "C40", "C31", "C43", "C22"],'genes':["Opn4", "Eomes", "Tbx20", "Spp1", "Nmb", "Il1rapl2", "Serpine2"]},
           'F-RGCs': {'cluster_names':["C4", "C38", "C28", "C32"],'genes':["Foxp2", "Foxp1", "Pou4f1","Pou4f2","Pou4f3","Ebf3","Irx4","Pde1a","Anxa3","Cdk15","Coch"]},
           'T-RGCs': {'cluster_names':["C5", "C17", "C21", "C42", "C9"],'genes':["Tbr1","Jam2","Pou4f3","Pou4f2","Calb2","Pcdh20","Irx4","Calca","Spp1","Plpp4"]},
           'alpha-RGCs': {'cluster_names':["C43","C41","C45","C42"],'genes':["Spp1","Calb1","Pou4f1","Pou4f3","Opn4","Il1rapl2","Kit","Tpbg","Fes"]}
          }
    DOTPLOT_CMAP='BuPu'
    MEDIUM_FIG_SIZE = (6.4*2,4.8*2)






    def __init__(self,runs,base_dir,regions_lst=[0,1,2,3],
                im_script_dir= "imaging_scripts", subfolder="comparative_figures",data_f="data",merged_f="Cellpose_merged.h5ad",merged_remapped_f="Cellpose_merged_remapped.h5ad",all_reassigned_cells_f="all_cells_reassigned.h5ad",RGC_cache_f="RGC_only.h5ad",
                final_f="Cellpose_final_assignments.h5ad",stitched_m_f="stitched_masks_c2knl_full_retina_comb.npy",feature_names_f="models/150g/feature_names.pickle", xgb_model_f="models/150g/xgb_150g",rgc_median_transcripts=176,postfix=None):# runs=[140g_rn3, 140g_rn4...]
        self.runs=runs
        self.merged_f = self._add_postfix(merged_f,postfix)
        self.merged_remapped_f =  self._add_postfix(merged_remapped_f,postfix)
        self.reassigned_cells_f = self._add_postfix(all_reassigned_cells_f,postfix)
        self.RGC_cache_f = self._add_postfix(RGC_cache_f,postfix)
        self.final_f = self._add_postfix(final_f,postfix)
        self.base_dir = base_dir

        self.data_dir = os.path.join(base_dir,data_f)
        self.im_script_dir=  os.path.join(os.getcwd(), im_script_dir)
        self.removal_areas_per_region = rc.BIPOLAR_ZONES
        self.subfolder = subfolder
        self.regions_lst = regions_lst.copy()
        self.stitched_m_f = stitched_m_f
        self.feature_names_f=feature_names_f
        self.xgb_model_f=xgb_model_f
        self.rgc_median_transcripts=rgc_median_transcripts
    
    
    """Just adds postfix to file names in case I want to try processing steps on the same run (without overriding previous ones)"""
    def _add_postfix(self,name,postfix=None):
        if postfix is not None:
            name = name.replace(".h5ad", f"{postfix}.h5ad")
        return name
    
    """Take a run and move the data associated with a different processing version (e.g. different xgboost model) to its own "run" folder using the postfix to identify the run
    Args:
        postfix (str): String of postfix e.g. (postfix="_130")
        run (str): E.g. 140g_rn3
        run_new (str): name of new folder e.g. "140g_rn3_130"
    Writes:
        All datafiles with postfidx under directory run_new
    """
    def clone_run(self,postfix,run,run_new):
        assert run in self.runs
        files_to_copy=[self.merged_f,self.merged_remapped_f,self.reassigned_cells_f,self.RGC_cache_f,self.final_f,"train_dict.pickle"]
        data_f = os.path.join(self.data_dir,run)
        out_f = os.path.join(self.data_dir,run_new)
        for f in files_to_copy:
            f_stripped = f.replace(f"{postfix}.h5ad",".h5ad")
            src = os.path.join(data_f,f)
            if not os.path.isdir(out_f):
                os.mkdir(out_f)
            dst = os.path.join(out_f,f_stripped)
            shutil.copy(src,dst)

    """Merge data across several runs into one anndata
    Args:
        runs (lst): Run names corresponding to anndata objects you want to merge into a single anndata
        adata_fout (str): Relative path+filename in data directory to store anndata
    Modifies:
    Writes:
        merged_ann (anndata): Merged anndata object
    """

    def merge_runs(self,runs,subd="control", adata_fout="merged_control.h5ad", keys=None):
        ann_arr= []
        if keys is None:
            pattern = re.compile(r'.*rn([0-9]*)')
            keys = [int(pattern.match(run).group(1)) for run in runs]

        for run in runs:
            adata=sc.read_h5ad(os.path.join(self.data_dir,run,self.final_f))
            ann_arr.append(adata)

        merged_ann = ad.concat(ann_arr,axis=0,label='run', index_unique='_rn', keys =keys)
        output_dir = os.path.join(self.data_dir,subd)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        merged_ann.write_h5ad(os.path.join(output_dir, adata_fout))

        return merged_ann
    
    def cluster_all(self,adata_f="merged_control.h5ad",subd="control",adata_fout="merged_control_cluster.h5ad",n_neighbors=70,batch_key=None):
        fpath = os.path.join(self.data_dir,subd,adata_f)
        adata=sc.read_h5ad(fpath)
        print(adata)
        adata=adata[adata.obs['to_keep'] & adata.obs['pass_cutoff']]
        adata = pr.rearrange_and_preprocess(adata,feature_names_f=self.feature_names_f,target_sum=self.rgc_median_transcripts, batch_key=batch_key,use_rep='X')
        pr.cluster(adata,n_neighbors=n_neighbors,resolution=1,key_added='joint_leiden',use_rep='X')
        adata.write_h5ad(os.path.join(self.data_dir,subd,adata_fout))
        return adata
    
    
    """Run classification pipeline on only RGC's in merged data object
    Args:
        adata_f (str): anndata file containing merged data object with all cells
        subd (str): directory below data to save files in
        adata_fout (str): Output anndata file to save in subd
        batch_key (str): obs.column to do batch correction with
        cutoff (float) Cutoff to use for xgboost
        n_neighbors (int):  Number of neighbors to use for neighborhood graph
        resolution (float): Resolution to cluster with

    """
    def joint_RGC_pipeline(self,adata_f="merged_control.h5ad",subd="control", adata_fout="RGC_control.h5ad", class_col='class',batch_key=None, cutoff=0.5,n_neighbors=50,resolution=1.0):
        fpath = os.path.join(self.data_dir,subd,adata_f)
        print("Loading from: ", fpath)
        print("Will output to: ",os.path.join(self.data_dir,subd, adata_fout))
        adata = sc.read_h5ad(fpath)
        adata=adata[adata.obs[class_col]=='RGC']
        adata = pr.generate_RGC_classifications(
            adata, 
            feature_names_f=self.feature_names_f,
            num_classes=constants.NUM_RGC_TYPES, 
            xgb_model_f=self.xgb_model_f, 
            unassigned_indicator=constants.NUM_RGC_TYPES, 
            target_sum=self.rgc_median_transcripts,
            cutoff=cutoff,
            batch_key=batch_key,
            n_neighbors=n_neighbors,
            resolution=resolution
        )
        adata.write_h5ad(os.path.join(self.data_dir,subd, adata_fout))
        return adata

    """
    Merge across regions; Takes input run and merges across region
    """
    # Merge across regions
    def merge_and_remap(self,run): #TODO: Shift all writes to the aggregate pipeline class instead
        if run in self.ALIGNMENT_MAP:
            alignment = self.ALIGNMENT_MAP[run]
        else:
            alignment=None
        stain = "DAPI"

        IM_DIR = os.path.join(self.base_dir, "imaging_scripts", "images") # In this case the images are "data"
        print(IM_DIR)

        subdirectories = [f"{run}_rg{i}" for i in self.regions_lst] #TODO: Update this so that each run is associaed with regions (e.g. dictionary)
        a = ap.AggregatePipeline(subdirectories,self.removal_areas_per_region,feature_names_f=self.feature_names_f,xgb_model_f=self.xgb_model_f)
        a.merge_region_dfs(alignment,stain,IM_DIR,keys=self.regions_lst,merged_f=self.merged_f)  

        df=a.merged_ann.obs
        filt=df['to_keep'] & df['pass_cutoff']
        adata_f = a.create_group_map(filt=filt,subd=run, is_region=False, n_neighbors=60, output_f=self.reassigned_cells_f)
        sc.pl.umap(adata_f,color=['Rbpms','Slc17a6','Pou4f1','Pou4f2'],layer='raw',vmin=0,vmax=10)
        sc.pl.umap(adata_f,color=['Chat','Tfap2a','Tfap2b'],layer='raw',vmin=0,vmax=10)
        sc.pl.umap(adata_f,color='counts',vmin=15,vmax=400)

        

        a.merged_ann.write(os.path.join(self.data_dir,run,self.merged_remapped_f))
        return a
    
    def apply_classifier(self,run, info,cluster_col='leiden'):
        subdirectories = [f"{run}_rg{i}" for i in self.regions_lst]
        a = ap.AggregatePipeline(subdirectories,self.removal_areas_per_region,feature_names_f=self.feature_names_f,xgb_model_f=self.xgb_model_f)
        model_dir=info['model_dir']
        model_name=f"valid_{info['model_name']}"
        train_dict_f=info['train_dict_name']
        run=run
        adata=sc.read_h5ad(os.path.join(self.data_dir,run,self.reassigned_cells_f))
        neuronal_classes = {"Bipolar cell","GabaAC",'GlyAC','Neither Gaba nor Gly AC','RGC'}
        non_neuronal_classes = {"Endothelial","Horizontal cell",'Microglia','Muller glia','Pericyte'}
        adata.obs['class'] = adata.obs['mapped_group'].map(a.GROUP_MAP)


        cutoff=0.5
        cl.apply_model(
            adata=adata,
            model_f=os.path.join(model_dir,model_name),
            train_dict=os.path.join(model_dir,train_dict_f),
            cutoff=cutoff,
            output_dir="",
            output_f="", 
            unassigned_indicator='Unassigned', 
            prefix='', 
            test_key=cluster_col,
            train_key_added='retina_class',
            prefix_key=None,
            xlabel='Predicted',
            ylabel='Clustering'
        ) #Test key is usually clustering

        adata.obs['superclass'] = ['Neuron' if x in neuronal_classes else 'Non-neuron' if x in non_neuronal_classes else 'Unassigned' for x in adata.obs['retina_class']]

        confusion_s=pd.crosstab(adata.obs[cluster_col], adata.obs['superclass'])
        confusion_m = pd.crosstab(adata.obs[cluster_col],adata.obs['class'])

        for c in np.unique(adata.obs['leiden']):
            s = confusion_s.loc[c]
            s=s/np.sum(s)
            m = confusion_m.loc[c]
            m=m/np.sum(m)
            if m['Other']>0.5:
                if s['Non-neuron']+s['Unassigned']>0.7:
                    print(f"Remap {c} to be non-neuronal")
                    adata.obs.loc[adata.obs['leiden']==c,'class']='Non-neuronal'
                    
        adata.obs.loc[adata.obs['class']=='Other','class']='Low quality neuronal'

        merged_ann=sc.read_h5ad(os.path.join(self.data_dir,run,self.merged_remapped_f))
        if 'class' in merged_ann.obs.columns:#In case you are rerunning this code since it reads and writes to same file name
            merged_ann.obs.drop(columns='class',inplace=True)
        merged_ann.obs = merged_ann.obs.merge(adata.obs[['class']],on='cell_id',how='left')# Add leiden clustering for later remapping
        merged_ann.obs['class'] = merged_ann.obs['class'].fillna(value='N/A')
        merged_ann.write_h5ad(os.path.join(self.data_dir,run,self.merged_remapped_f))
        print(os.path.join(self.data_dir,run,self.merged_remapped_f))
        return adata,merged_ann

    
    def cache_RGCs(self,run,input_train_dict_f,train_dict_f="train_dict.pickle",cutoff=0.5):
        assert run in self.runs
        adata=sc.read_h5ad(os.path.join(self.data_dir,run,self.merged_remapped_f))
        print(os.path.join(self.data_dir,run,self.merged_remapped_f))
        subdirectories = [f"{run}_rg{i}" for i in self.regions_lst]
        a = ap.AggregatePipeline(subdirectories,self.removal_areas_per_region,feature_names_f=self.feature_names_f,xgb_model_f=self.xgb_model_f)
        a.merged_ann = adata

        with open(input_train_dict_f,'rb') as f:
            a.train_dict = pickle.load(f)

        a.generate_RGC_classifications(cutoff=cutoff)
        a.adata_cache.write(os.path.join(self.data_dir,run,self.RGC_cache_f))
        with open(os.path.join(self.data_dir,run,train_dict_f), "wb") as f:
            pickle.dump(a.train_dict, f)
    
    def create_final_assignments(self,run):
        adata=sc.read_h5ad(os.path.join(self.data_dir,run,self.merged_remapped_f))
        print(adata)
        adata_RGC=sc.read_h5ad(os.path.join(self.data_dir,run,self.RGC_cache_f))
        subdirectories = [f"{run}_rg{i}" for i in self.regions_lst]
        a = ap.AggregatePipeline(subdirectories,self.removal_areas_per_region,feature_names_f=self.feature_names_f,xgb_model_f=self.xgb_model_f)
        a.merged_ann = adata
        a.adata_cache = adata_RGC
        a.train_dict = self._load_train_dict(run)
        a.create_final_assignments(type_col='cluster', class_col='class', new_col='final_assignment')
               
        a.merged_ann.write(os.path.join(self.data_dir,run,self.final_f))
        return a.merged_ann
        
    def plot_confusion_matrix(self,run,train_key='joint_type',test_key='leiden',train_dict_f="train_dict.pickle",re_order_cols=None,figure_dir=None, fname=None):
        assert run in self.runs
        adata = sc.read_h5ad(os.path.join(self.data_dir,run,self.RGC_cache_f))
        subdirectories = [f"{run}_rg{i}" for i in self.regions_lst]
        a = ap.AggregatePipeline(subdirectories,self.removal_areas_per_region)
        a.adata_cache = adata
        a.class_train_dict = self._load_train_dict(run)
        if figure_dir is not None and fname is not None:
            save_f = os.path.join(figure_dir,fname)
        a.plot_confusion_matrix(re_order_cols=re_order_cols,train_key=train_key,test_key=test_key,save_f=save_f)
    
    def plot_confusion_matrix_v2(self,run,train_key='joint_type',test_key='leiden',train_dict_f="train_dict.pickle",c='blue', class_train_dict=None, class_test_dict=None, re_order_cols=None,figure_dir=None, fname=None):
        assert run in self.runs
        adata = sc.read_h5ad(os.path.join(self.data_dir,run,self.RGC_cache_f))
        subdirectories = [f"{run}_rg{i}" for i in self.regions_lst]
        class_train_dict = self._load_train_dict(run)
        if class_train_dict is None:
            class_train_dict = self._load_train_dict(run)
        if class_test_dict is None:
            num_clusters = len(np.unique(adata.obs[test_key]))
            class_test_dict = {f"M{i}":int(i) for i in range(num_clusters)}

        output_f= os.path.join(figure_dir,fname)
        with plt.rc_context({"font.size":16, "figure.figsize":[20,20],"figure.dpi":300}):
            pu.plot_mapping_new(test_labels=adata.obs[test_key].values.astype(int), #y-axis
                            test_predlabels=adata.obs[train_key].values,#x-AXIS
                            test_dict=class_test_dict,  # y-axis
                            train_dict=class_train_dict, # x-axis
                            re_order=True,
                            re_order_cols=re_order_cols,
                            re_index=True,
                            xaxislabel='scRNA-seq', yaxislabel='MERFISH Leiden Cluster',
                            c=c,
                            save_as=output_f
            )
    
    def _load_train_dict(self,run, train_dict_f="train_dict.pickle"):
        with open(os.path.join(self.data_dir,run,train_dict_f), "rb") as f:
            train_dict= pickle.load(f)
        return train_dict


    
    def plot_run_distributions(self,run_x,run_y,xlabel,ylabel,type_col_x='cluster',type_col_y='group_num',figure_dir="comparative_figures",fname="type_distribution_scatter.png",color='blue'):
        assert (run_x in self.runs or run_x==self.ATLAS_RUN) and (run_y in self.runs or run_y==self.ATLAS_RUN)

        if run_x==self.ATLAS_RUN:
            adata_x = sc.read_h5ad(self.RGC_ATLAS_F)
        else:
            adata_x = sc.read_h5ad(os.path.join(self.data_dir,run_x,self.RGC_cache_f))

        if run_y==self.ATLAS_RUN:
            adata_y = sc.read_h5ad(self.RGC_ATLAS_F)
        else:
            adata_y = sc.read_h5ad(os.path.join(self.data_dir,run_y,self.RGC_cache_f))
        print(fname)
        pu.plot_type_dist_scatter(
            adata_x,
            adata_y,
            title="",
            xlabel=xlabel,
            ylabel=ylabel,
            type_col_x=type_col_x,
            type_col_y=type_col_y,
            figure_dir=figure_dir,
            fname=fname,
            color='blue'
        )
        del adata_x,adata_y
        gc.collect()
    

    def _preprocess_bae_data(self,data_dir=os.path.join("data","bae"),output_f='gc_info'):
        cell_info = scipy.io.loadmat(os.path.join(data_dir,constants.BAE_CELL_INFO),simplify_cells=True)
        gc_list = scipy.io.loadmat(os.path.join(data_dir,constants.BAE_GC_LIST),simplify_cells=True)


        gc_info = []
        for cell in cell_info['cell_info']:
            if cell['cell_id'] in gc_list['gc_list']:
                gc_info.append(cell)
        gc_info = pd.DataFrame(gc_info) 
        to_duplicate = gc_info[gc_info['type']=='1ws'].copy()
        to_duplicate['type']= '1ws_duplicate'
        gc_info = pd.concat([to_duplicate,gc_info])
        gc_info['final_assignment'] = gc_info['type'].map({v:k for k,v in constants.tran_bae_mapping.items()})
        gc_info.to_csv(os.path.join(data_dir,output_f))
        return gc_info


    
    def plot_bae_size_comparison(self,runs,max_area_col='max_areas',type_col='final_assignment', ylim=(0,3000),ylabel=r'Soma Volume ($\mu m^3$)',xlabel='RGC Type', remake=False, figsize=(8,20),data_dir=os.path.join("data","bae"),output_f='gc_info',figure_dir=os.path.join("comparative_figures",'control_130'),fname="compare_size_dist_bae.png"):
        df_arr=[]
        if os.path.exists(os.path.join(data_dir,output_f)) and not remake:
            gc_info = pd.read_csv(os.path.join(data_dir,output_f))
        else:
            gc_info = self._preprocess_bae_data(data_dir=data_dir, output_f=output_f)
        gc_info['Method'] = 'EM'

        for i,run in enumerate(runs):
            assert run in self.runs
            adata= sc.read_h5ad(os.path.join(self.data_dir,run,self.final_f))
            # to_keep = np.unique(gc_info['final_assignment'].dropna())
            # adata.obs.index = adata.obs.index + f"_{run}"
            # df = adata.obs[adata.obs[type_col].isin(to_keep)]
            # df[type_col] = df[type_col].cat.remove_unused_categories()
            df = adata.obs
            df_arr.append(df)
        df_joint = pd.concat(df_arr,axis=0,ignore_index=True)
        df_joint['Method'] = 'MERFISH'
        df_joint['soma_size']=4/3*np.pi*(np.sqrt(df_joint[max_area_col]/(np.pi)))**3

        # Add the bae et al data and plot
        df_joint = pd.concat([df_joint,gc_info],ignore_index=True)
        """
        with plt.rc_context({"figure.figsize":figsize, "figure.dpi":300, "font.sans-serif":'Arial',"font.family":"sans-serif","font.size":24}):
            sns.boxplot(data=df_joint,x='soma_size', y='final_assignment',hue='Method', order=order,fliersize=0)
            plt.xlim(xlim)
            plt.xticks(np.arange(0, 3500, 1000))
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(bbox_to_anchor=(0.5,1.0))
            plt.savefig(os.path.join(figure_dir,fname),bbox_inches='tight')
        plt.show()
        """
        RGC_types = [f'C{i}' for i in range(1,46)]
        df_joint = df_joint[df_joint['final_assignment'].isin(RGC_types)]
        order = df_joint.groupby(by='final_assignment')['soma_size'].median().sort_values(ascending=False).index
        print(df_joint['final_assignment'].unique())
        with plt.rc_context({"figure.figsize":figsize, "figure.dpi":300, "font.sans-serif":'Arial',"font.family":"sans-serif","font.size":8}):
            sns.boxplot(data=df_joint,y='soma_size', x='final_assignment',hue='Method', palette={'MERFISH':'cornflowerblue','EM':'lightcoral'}, order=order,fliersize=0)
            plt.ylim(ylim)
            plt.yticks(np.arange(0, 3500, 1000))
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.xticks(rotation=90)
            plt.legend(bbox_to_anchor=(0.5,1.0))
            plt.savefig(os.path.join(figure_dir,fname),bbox_inches='tight')
        plt.show()
        
        return df_joint


    


    """Return proportions of cell types in anndata;
        #TODO: Wrap these functions?
    """
    def _extract_scatter_info(self,adata,type_col,unassigned_indicator=45):
        tmp= adata[adata.obs[type_col] != unassigned_indicator]
        type_counts =  np.bincount(tmp.obs[type_col],minlength=unassigned_indicator)
        return type_counts/np.sum(type_counts)*100
    
    def plot_run_distributions_v2(self,runs_x,runs_y,xlabel,ylabel,type_col_x='cluster',type_col_y='group_num',figure_dir="comparative_figures",fontsize=48,fname="type_distribution_scatter.png",dot_color='blue',err_color='black',line_color='blue',s=30):
        prop_counts_x=np.zeros((len(runs_x), constants.NUM_RGC_TYPES))
        for i,run_x in enumerate(runs_x):
            if run_x==self.ATLAS_RUN:
                adata = sc.read_h5ad(self.RGC_ATLAS_F)
            else:
                adata = sc.read_h5ad(os.path.join(self.data_dir,run_x,self.RGC_cache_f))
            prop_counts_x[i] = self._extract_scatter_info(adata=adata,type_col=type_col_x)

        prop_counts_y=np.zeros((len(runs_y), constants.NUM_RGC_TYPES))
        for i,run_y in enumerate(runs_y):
            if run_y==self.ATLAS_RUN:
                adata = sc.read_h5ad(self.RGC_ATLAS_F)
            else:
                adata = sc.read_h5ad(os.path.join(self.data_dir,run_y,self.RGC_cache_f))
            prop_counts_y[i] = self._extract_scatter_info(adata=adata,type_col=type_col_y)
        
        mean_y, yerr = np.mean(prop_counts_y,axis=0), np.std(prop_counts_y,axis=0)
        mean_x, xerr = np.mean(prop_counts_x,axis=0), np.std(prop_counts_x,axis=0)
        context = {'font.sans-serif':"Arial",'font.family':"sans-serif","figure.dpi":200}
        print(fontsize)
        with plt.rc_context(context):
            fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6,6),dpi=200)
            ax.set_ylabel(ylabel,fontsize=fontsize)
            ax.set_xlabel(xlabel,fontsize=fontsize)

            max_x=9
            corr_matrix = np.corrcoef(mean_x,mean_y)
            r = corr_matrix[0,1]
            plt.text(5, 7, 'r = %0.2f' % r,fontsize=fontsize*2/3)

            ax.set_xlim(0,max_x)
            ax.set_ylim(0,max_x)
            x_y_l = np.linspace(0,9,num=50)
            ax.errorbar(x=mean_x,y=mean_y,yerr=yerr,xerr=xerr,linestyle='none',color=err_color,capsize=3.0,zorder=0)
            ax.plot(x_y_l,x_y_l,linewidth=1,linestyle='dotted',color=line_color)
            ax.scatter(mean_x,mean_y,s=s,color=dot_color,zorder=1)
            fig.savefig(fname=f'{figure_dir}/{fname}',bbox_inches="tight")
            fig.show()
    
    def plot_run_distributions_v3(self,runs_x,runs_y,xlabel,ylabel,type_col_x='cluster',type_col_y='group_num',figure_dir="comparative_figures",fontsize=48,fname="type_distribution_scatter.png",dot_color='blue',err_color='black',line_color='blue',s=30):
        prop_counts_x=np.zeros((len(runs_x), constants.NUM_RGC_TYPES))
        for i,run_x in enumerate(runs_x):
            if run_x==self.ATLAS_RUN:
                adata = sc.read_h5ad(self.RGC_ATLAS_F)
            else:
                adata = sc.read_h5ad(os.path.join(self.data_dir,run_x,self.RGC_cache_f))
            prop_counts_x[i] = self._extract_scatter_info(adata=adata,type_col=type_col_x)

        prop_counts_y=np.zeros((len(runs_y), constants.NUM_RGC_TYPES))
        for i,run_y in enumerate(runs_y):
            if run_y==self.ATLAS_RUN:
                adata = sc.read_h5ad(self.RGC_ATLAS_F)
            else:
                adata = sc.read_h5ad(os.path.join(self.data_dir,run_y,self.RGC_cache_f))
            prop_counts_y[i] = self._extract_scatter_info(adata=adata,type_col=type_col_y)
        
        mean_y, yerr = np.mean(prop_counts_y,axis=0), np.std(prop_counts_y,axis=0)
        mean_x, xerr = np.mean(prop_counts_x,axis=0), np.std(prop_counts_x,axis=0)
        context = {'font.sans-serif':"Arial",'font.family':"sans-serif","figure.dpi":200}
        print(fontsize)
        with plt.rc_context(context):
            fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6,6),dpi=200)
            ax.set_ylabel(ylabel,fontsize=fontsize)
            ax.set_xlabel(xlabel,fontsize=fontsize)

            corr_matrix = np.corrcoef(mean_x,mean_y)
            r = corr_matrix[0,1]
            plt.text(3, 7, 'r = %0.2f' % r,fontsize=fontsize*2/3)

            x_y_l = np.linspace(0,9,num=50)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.errorbar(x=mean_x,y=mean_y,yerr=yerr,xerr=xerr,linestyle='none',color=err_color,capsize=3.0,zorder=0)
            ax.plot(x_y_l,x_y_l,linewidth=1,linestyle='dotted',color=line_color)
            ax.scatter(mean_x,mean_y,s=s,color=dot_color,zorder=1)
            fig.savefig(fname=f'{figure_dir}/{fname}',bbox_inches="tight")
            fig.show()



            
    def plot_cluster_frequency_bars(self,run_1,run_2,label_1,label_2,type_col_1='cluster',type_col_2='cluster',figure_dir="comparative_figures/control",fname="type_distribution_bar.png"):
        assert (run_1 in self.runs or run_1==self.ATLAS_RUN) and (run_2 in self.runs or run_2==self.ATLAS_RUN)
        train_dict = None
        if run_1==self.ATLAS_RUN:
            adata_1 = sc.read_h5ad(self.RGC_ATLAS_F)
        else:
            adata_1 = sc.read_h5ad(os.path.join(self.data_dir,run_1,self.RGC_cache_f))
            train_dict = self._load_train_dict(run_1)

        if run_2==self.ATLAS_RUN:
            adata_2 = sc.read_h5ad(self.RGC_ATLAS_F)
        else:
            adata_2 = sc.read_h5ad(os.path.join(self.data_dir,run_2,self.RGC_cache_f))
            train_dict = self._load_train_dict(run_2)
        
        unassigned_indicator = len(train_dict)-1 # E.g. 46 entries-> '45' is last entry and is unassigned
        print(adata_1.var)
        pu.plot_cluster_frequency_bars(
            adata_1=adata_1,
            adata_2=adata_2,
            ck_1=type_col_1,
            ck_2=type_col_2,
            label_1=label_1,
            label_2=label_2, 
            train_dict=train_dict,
            unassigned_indicator=unassigned_indicator,
            figure_dir=figure_dir,
            fname=fname
        )
    
    
    
    def plot_gene_expression(self,run_x,run_y,xlabel,ylabel,type_col_x='cluster',type_col_y='cluster',figure_dir="comparative_figures",fname="gene_expression_comparison.png"):
        train_dict = None
        if run_x==self.ATLAS_RUN:
            adata_x = sc.read_h5ad(self.RGC_ATLAS_F)
            adata_x.var.rename(index=self.RENAMED_GENE_MAP,inplace=True)
        else:
            adata_x = sc.read_h5ad(os.path.join(self.data_dir,run_x,self.RGC_cache_f))
            train_dict = self._load_train_dict(run_x)

        if run_y==self.ATLAS_RUN:
            adata_y = sc.read_h5ad(self.RGC_ATLAS_F)
            adata_y.var.rename(columns=self.RENAMED_GENE_MAP,inplace=True)
        else:
            adata_y = sc.read_h5ad(os.path.join(self.data_dir,run_y,self.RGC_cache_f))
            train_dict = self._load_train_dict(run_y)

        # Note, could bug if train dict for both runs are different...but they will be the same
        inv_train_dict = {v:k for k,v in train_dict.items()}
        adata_x.obs['cluster_names'] = adata_x.obs[type_col_x].map(inv_train_dict)
        adata_y.obs['cluster_names'] = adata_y.obs[type_col_x].map(inv_train_dict)

     
        for group in self.TRAN_FIG_MAPPING:
            key = self.TRAN_FIG_MAPPING[group]       
            adata_xg = adata_x[adata_x.obs['cluster_names'].isin(key['cluster_names'])]
            adata_yg = adata_y[adata_y.obs['cluster_names'].isin(key['cluster_names'])]
            var_names = key['genes']
            categories = key['cluster_names']
            print(var_names)
            print(categories)

            pu.plot_gene_expression(
                adata_x=adata_xg,
                adata_y=adata_yg,
                xlabel=f"{xlabel}: {group}",
                ylabel=f"{ylabel}: {group}",
                var_names=var_names,
                categories=categories,
                typename_col_x='cluster_names',
                typename_col_y='cluster_names',
                figure_dir=figure_dir,
                fname=f"{fname}_{group}.png",
                cmap=self.DOTPLOT_CMAP
            )
    
    def plot_gene_expression(self,run_x,run_y,xlabel,ylabel,type_col_x='cluster',type_col_y='cluster',figure_dir="comparative_figures",fname="gene_expression_comparison.png"):
        train_dict = None
        if run_x==self.ATLAS_RUN:
            adata_x = sc.read_h5ad(self.RGC_ATLAS_F)
            adata_x.var.rename(index=self.RENAMED_GENE_MAP,inplace=True)
        else:
            adata_x = sc.read_h5ad(os.path.join(self.data_dir,run_x,self.RGC_cache_f))
            train_dict = self._load_train_dict(run_x)

        if run_y==self.ATLAS_RUN:
            adata_y = sc.read_h5ad(self.RGC_ATLAS_F)
            adata_y.var.rename(columns=self.RENAMED_GENE_MAP,inplace=True)
        else:
            adata_y = sc.read_h5ad(os.path.join(self.data_dir,run_y,self.RGC_cache_f))
            train_dict = self._load_train_dict(run_y)

        # Note, could bug if train dict for both runs are different...but they will be the same
        inv_train_dict = {v:k for k,v in train_dict.items()}
        adata_x.obs['cluster_names'] = adata_x.obs[type_col_x].map(inv_train_dict)
        adata_y.obs['cluster_names'] = adata_y.obs[type_col_y].map(inv_train_dict)

     
        for group in self.TRAN_FIG_MAPPING:
            key = self.TRAN_FIG_MAPPING[group]       
            adata_xg = adata_x[adata_x.obs['cluster_names'].isin(key['cluster_names'])]
            adata_yg = adata_y[adata_y.obs['cluster_names'].isin(key['cluster_names'])]
            var_names = key['genes']
            categories = key['cluster_names']
            print(var_names)
            print(categories)

            pu.plot_gene_expression(
                adata_x=adata_xg,
                adata_y=adata_yg,
                xlabel=f"{xlabel}: {group}",
                ylabel=f"{ylabel}: {group}",
                var_names=var_names,
                categories=categories,
                typename_col_x='cluster_names',
                typename_col_y='cluster_names',
                figure_dir=figure_dir,
                fname=f"{fname}_{group}.png",
                cmap=self.DOTPLOT_CMAP
            )
    
    def plot_tran_gene_expression_v2(self,runs, xlabel,type_col_x='cluster',type_col_y='cluster',figure_dir="comparative_figures",fname="gene_expression_comparison.png"):
        adata_arr= []
        for run in runs:
            assert run in self.runs
            adata_x = sc.read_h5ad(os.path.join(self.data_dir,run,self.RGC_cache_f))
            adata_arr.append(adata_x)
            train_dict = self._load_train_dict(run)

        adata_x = ad.concat(adata_arr)
        adata_y = sc.read_h5ad(self.RGC_ATLAS_F)
        adata_y.var.rename(index=self.RENAMED_GENE_MAP,inplace=True)
        print(adata_y)

        

        # Note, could bug if train dict for both runs are different...but they will be the same
        inv_train_dict = {v:k for k,v in train_dict.items()}
        adata_x.obs['cluster_names'] = adata_x.obs[type_col_x].map(inv_train_dict)
        adata_y.obs['cluster_names'] = adata_y.obs[type_col_y].map(inv_train_dict)

     
        for group in self.TRAN_FIG_MAPPING:
            key = self.TRAN_FIG_MAPPING[group]       
            adata_xg = adata_x[adata_x.obs['cluster_names'].isin(key['cluster_names'])]
            adata_yg = adata_y[adata_y.obs['cluster_names'].isin(key['cluster_names'])]
            var_names = key['genes']
            categories = key['cluster_names']

            ylabel='scRNA-seq'

            pu.plot_gene_expression(
                adata_x=adata_xg,
                adata_y=adata_yg,
                xlabel=f"{xlabel}: {group}",
                ylabel=f"{ylabel}: {group}",
                var_names=var_names,
                categories=categories,
                typename_col_x='cluster_names',
                typename_col_y='cluster_names',
                figure_dir=figure_dir,
                fname=f"{fname}_{group}.png",
                cmap=self.DOTPLOT_CMAP
            )
    
    def plot_gene_expression_v2(self,runs, xlabel,var_names, group, categories=None, type_col_x='cluster',type_col_y='cluster',figure_dir="comparative_figures",fname="gene_expression_comparisond.png"):
        adata_arr= []
        for run in runs:
            assert run in self.runs
            adata_x = sc.read_h5ad(os.path.join(self.data_dir,run,self.RGC_cache_f))
            adata_arr.append(adata_x)
            train_dict = self._load_train_dict(run)

        adata_x = ad.concat(adata_arr)
        adata_y = sc.read_h5ad(self.RGC_ATLAS_F)
        adata_y.var.rename(index=self.RENAMED_GENE_MAP,inplace=True)
        print(adata_y)

        

        # Note, could bug if train dict for both runs are different...but they will be the same
        inv_train_dict = {v:k for k,v in train_dict.items()}
        adata_x.obs['cluster_names'] = adata_x.obs[type_col_x].map(inv_train_dict)
        adata_x=adata_x[adata_x.obs['cluster_names'] != 'Unassigned']
        adata_y.obs['cluster_names'] = adata_y.obs[type_col_y].map(inv_train_dict)


        if categories is not None:
            adata_x = adata_x[adata_x.obs['cluster_names'].isin(categories)]
            adata_y =  adata_y[adata_y.obs['cluster_names'].isin(categories)]

        ylabel='scRNA-seq'

        pu.plot_gene_expression_v2(
            adata_x=adata_x,
            adata_y=adata_y,
            xlabel=f"{xlabel}: {group}",
            ylabel=f"{ylabel}: {group}",
            var_names=var_names,
            categories=categories,
            typename_col_x='cluster_names',
            typename_col_y='cluster_names',
            figure_dir=figure_dir,
            fname=f"{fname}_{group}",
            cmap=self.DOTPLOT_CMAP
        )

    def plot_classification_threshold(self,run):
        assert run in self.runs
        adata = sc.read_h5ad(os.path.join(self.data_dir,run,self.RGC_cache_f))
        subdirectories = [f"{run}_rg{i}" for i in self.regions_lst]
        a = ap.AggregatePipeline(subdirectories,self.removal_areas_per_region)
        a.adata_cache = adata
        a.class_train_dict = self._load_train_dict(run)
        a.plot_classification_threshold()
    
    
    def plot_classification_threshold_v2(self,runs,title="",legend_labels=None, figure_dir=os.path.join("comparative_figures","runs"),fname="Model_probability_comparisons.png"):
        # Compare cell type size distributions
        df_arr=[]
        for i,run in enumerate(runs):
            assert run in self.runs
            adata= sc.read_h5ad(os.path.join(self.data_dir,run,self.RGC_cache_f))
            class_train_dict = self._load_train_dict(run)
            ordering = [t[0] for t in (sorted(class_train_dict.items(),key=lambda t: t[1]))]# Create ordered list of keys in train_dict sorted using values of train_dict
            ordering.remove('Unassigned')

            y_probs = adata.obsm['y_probs']
            adata.obs['cluster_probs'] = np.max(y_probs, axis=1)
            adata.obs['cluster_no_threshold'] = (np.argmax(y_probs,axis=1) + 1).astype(str)
            adata.obs['cluster_no_threshold'] = adata.obs['cluster_no_threshold'].apply(lambda x: 'C' + x)
            adata.obs['cluster_no_threshold'] = pd.Categorical(adata.obs['cluster_no_threshold'],categories=ordering)
            if legend_labels is not None:
                adata.obs['run'] = legend_labels[i]
            else:
                adata.obs['run'] = run

            # adata.obs.boxplot(column="cluster_probs",by="cluster_no_threshold",figsize=(20,6),showfliers=False,flierprops={'markersize':2})
            df_arr.append(adata.obs)
        df_joint = pd.concat(df_arr,axis=0)
        context = {'font.sans-serif':"Arial",'font.family':"sans-serif","font.size":24,"figure.dpi":100,"legend.fontsize":14,"figure.figsize":(23,6)}
        with plt.rc_context(context):
            ax = sns.boxplot(data=df_joint,y='cluster_probs',x='cluster_no_threshold',hue='run',showfliers=False)
            fontsize=24
            
            

        
            ax.plot(np.arange(46) ,[0.5]*46,color='r',label='Model threshold',linestyle='dashed',alpha=0.7)
            ax.plot(np.arange(46) ,[0.02]*46,color='grey',label='Random assignment',linestyle='dashed',alpha=0.7)

            plt.suptitle('')
            ax.set_ylabel(r'Maximum probability assignment',fontsize=fontsize)
            ax.set_xlabel(r'RGC Type',fontsize=fontsize)
            ax.set_title(title)
            ax.grid(visible=False)
            ax.tick_params(axis='x',labelsize=fontsize,rotation=90)
            ax.tick_params(axis='y',labelsize=fontsize)
            plt.minorticks_off()
            plt.gca().spines['right'].set_color('none')
            plt.gca().spines['top'].set_color('none')
            plt.gca().xaxis.set_ticks_position('bottom')
            plt.gca().yaxis.set_ticks_position('left')
            plt.legend(bbox_to_anchor=(1.1,1.05))
            plt.title(title)
            plt.savefig(os.path.join(figure_dir,fname),dpi=200,facecolor='w',bbox_inches='tight')
            plt.show()
    
    def plot_types_by_region(self,run,group_name, cluster_names_included,fname,filt=True, RGC_only=True, other_names=None, col='cluster',s_group=5,s_other=0.1,colors='Red',invert_yaxis=True,invert_xaxis=False, map_names=True):
        assert run in self.runs
        if RGC_only:
            adata_f=self.RGC_cache_f
        else:
            adata_f=self.final_f

        adata= sc.read_h5ad(os.path.join(self.data_dir,run,adata_f))
        print(adata.obs)
        train_dict = self._load_train_dict(run)
        print(train_dict)
        adata.obs[col] = adata.obs[col].map({v:k for k,v in train_dict.items()})
        if filt and not RGC_only:
            adata=adata[adata.obs['to_keep'] & adata.obs['pass_cutoff']]
        cmap = plt.get_cmap('tab20')
        context = {'font.sans-serif':"Arial",'font.family':"sans-serif","font.size":14,"figure.dpi":100,"legend.fontsize":14}
        with plt.rc_context(context):
            df = adata.obs[~adata.obs[col].isin(cluster_names_included)]
            plt.scatter(df['aligned_center_x'], df['aligned_center_y'], label='Other types',s=s_other,color=cmap(15))
            for rg in np.unique(adata.obs['region']):
                df = adata.obs[(adata.obs['region'] ==rg) & (adata.obs[col].isin(cluster_names_included))]
                plt.scatter(df['aligned_center_x'],df['aligned_center_y'],label=rg,s=s_group)
            if invert_xaxis:
                plt.gca().invert_xaxis()
            if invert_yaxis:
                plt.gca().invert_yaxis()
            plt.title(group_name)
            plt.savefig(fname)
            plt.show()
    
    def plot_spatial_group(self,run,group_names, cluster_names_included,fname,filt=True, RGC_only=True, other_names=None, col='cluster',s_group=5,s_other=0.1,colors='Red',invert_yaxis=True,invert_xaxis=False, map_names=True):
        assert run in self.runs
        if RGC_only:
            adata_f=self.RGC_cache_f
        else:
            adata_f=self.final_f

        adata= sc.read_h5ad(os.path.join(self.data_dir,run,adata_f))
        if filt and not RGC_only:
            adata=adata[adata.obs['to_keep'] & adata.obs['pass_cutoff']]
        print(adata.obs['region'])
        subdirectories = [f"{run}_rg{i}" for i in self.regions_lst]
        a = ap.AggregatePipeline(subdirectories,self.removal_areas_per_region)
        a.class_train_dict = self._load_train_dict(run)
        if map_names:
            values = [a.class_train_dict[name] for name in cluster_names_included]
        else:
            values = cluster_names_included
        a.plot_spatial_group(adata=adata, values=values,group_names=group_names, other_names=other_names,col=col,fname=fname,s_group=s_group,s_other=s_other,colors=colors,invert_yaxis=invert_yaxis,invert_xaxis=invert_xaxis)
    
    """Black facecolor and scalebar"""
    def plot_spatial_group_v2(self,run,group_name, cluster_names_included=None,sup_dir="spatial_figures", fname="classes_scatter.png",leg_sep = True, filt=True, RGC_only=True, 
                              other_names=None, col='cluster',s_group=5,s_other=0.1,colors='Red',markerscale=10,draw_box=None,legend=True, invert_yaxis=True,invert_xaxis=False, map_names=True):
        assert run in self.runs
        if RGC_only:
            adata_f=self.RGC_cache_f
        else:
            adata_f=self.final_f

        adata= sc.read_h5ad(os.path.join(self.data_dir,run,adata_f))
        if filt and not RGC_only:
            adata=adata[adata.obs['to_keep'] & adata.obs['pass_cutoff']]
        # subdirectories = [f"{run}_rg{i}" for i in self.regions_lst]
        # a = ap.AggregatePipeline(subdirectories,self.removal_areas_per_region)
        # a.class_train_dict = self._load_train_dict(run)
        if cluster_names_included is None:
            cluster_names_included = adata.obs[col].unique()
        class_train_dict = self._load_train_dict(run)
        if map_names:
            values = [class_train_dict[name] for name in cluster_names_included]
        else:
            values = cluster_names_included

        context = {'font.sans-serif':"Arial",'font.family':"sans-serif","font.size":14,"figure.dpi":100,"legend.fontsize":14}
        with plt.rc_context(context):
            fig,ax=plt.subplots(nrows=1,ncols=1,figsize=self.MEDIUM_FIG_SIZE)
            df = adata[adata.obs[col].isin(values)].obs
            if type(colors) == str:
                colors = [colors]*len(values)
            elif type(colors[0]) == tuple:
                n_channels = len(colors[0])
                color_arr = np.zeros((df.shape[0],n_channels))
                m=dict(zip(values,colors))
                for i in range(n_channels):
                    color_arr[:,i] = df[col].apply(lambda x: m[x][i]).values
                colors=color_arr
            if leg_sep:
                for id in np.unique(df[col]):
                    df_p = df[df[col]==id]
                    ax.scatter(df_p['aligned_center_x'],df_p['aligned_center_y'],s=s_group,color=m[id],label=id)
            else:
                ax.scatter(df['aligned_center_x'],df['aligned_center_y'],s=s_group,color=colors,label=group_name)

            if other_names is not None:
                df = adata[~adata.obs[col].isin(values)].obs
                ax.scatter(df['aligned_center_x'],df['aligned_center_y'],s=s_other,c='Grey',label=other_names)
            
            if draw_box is not None:
                xl,xu,yl,yu = draw_box.reshape(-1)
                rect = patches.Rectangle((xl,yl),xu-xl,yu-yl,edgecolor='black',facecolor='none')
                ax.add_patch(rect)

            ax.set_facecolor('white')
            if legend:
                ax.legend(prop={'size':18}, markerscale=markerscale,frameon=False)
            ax.set_axis_off()
            if invert_xaxis:
                ax.invert_xaxis()
            if invert_yaxis:
                ax.invert_yaxis()
            scalebar = AnchoredSizeBar(ax.transData,
                            1000, '1 mm', 'lower center', 
                            pad=0.1,
                            color='black',
                            frameon=False,
                            size_vertical=1
                            )

            ax.add_artist(scalebar)
            
            fig.savefig(os.path.join(sup_dir, run,fname),dpi=300, bbox_inches='tight')
            fig.show()
    
    def plot_bar_counts(self,runs,labels,colors,col='class',values_to_exclude=None,filt=['to_keep','pass_cutoff'], map_group=None,figure_dir="comparative_figures",fname="compare_class_counts.png"):
        assert len(runs) == len(labels)
        counts_dict={}
        for i,run in enumerate(runs):
            data_f = os.path.join(self.data_dir,run,self.final_f)
            adata = sc.read_h5ad(data_f)
            df= adata.obs
            if values_to_exclude is not None:
                df = df[~df[col].isin(values_to_exclude)]
                df[col] = df[col].cat.remove_unused_categories()
                for cat in filt:
                    df = df[df[cat]]
            print("Counts:", df[col].value_counts(sort=True))
            if map_group is not None:
                group =df[col].apply(lambda x: x if x not in map_group else map_group[x])
            else:
                group = df[col]
            group_counts = group.value_counts().sort_index()
            index = group_counts.index
            print("Counts:\n", group_counts)
            counts_dict[labels[i]] = list(group_counts/len(df[col]))
        c = dict(zip(labels,colors))
        output_df = pd.DataFrame(counts_dict, index=index)
        output_df.plot.bar(color=c,edgecolor='black')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.ylabel("Proportion of cells",fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(os.path.join(figure_dir,fname),bbox_inches='tight')
        return adata

    def _get_counts(self,runs,col, filt, map_group=None, values_to_exclude=None,normalize=True):
        magnitudes=[]
        for i,run in enumerate(runs):
            data_f = os.path.join(self.data_dir,run,self.final_f)
            print(data_f)
            adata = sc.read_h5ad(data_f)
            df= adata.obs
            if values_to_exclude is not None:
                df = df[~df[col].isin(values_to_exclude)]
                df[col] = df[col].cat.remove_unused_categories()
            for cat in filt:
                df = df[df[cat]]
            if map_group is not None:
                group =df[col].apply(lambda x: x if x not in map_group else map_group[x])
            else:
                group = df[col]
            group_counts = group.value_counts().sort_index()
            index = group_counts.index
            if not normalize:
                magnitudes.append(np.array(group_counts))
            else:
                magnitudes.append(np.array(group_counts)/len(df[col]))
            
        return magnitudes,index
    
    def plot_bar_strips(self,runs,labels,ylabel,colors,col='class',values_to_exclude=None,filt=['to_keep','pass_cutoff'], map_group=None,figure_dir="comparative_figures/control",fname="compare_class_counts.png",normalize=True,figsize=(10,5)):
        return

    
    def plot_bar_counts_v2(self,runs_1,runs_2,label1,label2,ylabel, colors,col='class',values_to_exclude=None,filt=['to_keep','pass_cutoff'], map_group=None,figure_dir="comparative_figures/control",fname="compare_class_counts.png",normalize=True):
        counts_dict={}
        props_1,index_1 = self._get_counts(runs_1,col,filt,map_group=map_group, values_to_exclude=values_to_exclude,normalize=normalize)
        props_1=np.array(props_1)
        counts_dict[label1] = np.mean(props_1,axis=0)
        props_2,index_2 = self._get_counts(runs_2,col,filt,map_group=map_group, values_to_exclude=values_to_exclude,normalize=normalize)
        props_2=np.array(props_2)
        counts_dict[label2] = np.mean(props_2,axis=0)
        counts_dict['std1'] = np.std(props_1,axis=0)
        counts_dict['std2'] = np.std(props_2,axis=0)

        c = dict(zip([label1,label2],colors))
        output_df = pd.DataFrame(counts_dict, index=index_1)
        print(counts_dict)
        print(output_df)
        context = {'font.sans-serif':"Arial",'font.family':"sans-serif","font.size":24,"figure.dpi":200,"legend.fontsize":14}
        with plt.rc_context(context):
            ax = output_df[[label1, label2]].plot(kind='bar', color=colors,yerr=output_df[['std1','std2']].values.T,capsize=2.0,error_kw=dict(ecolor='k'))
            # x=ax.get_xticks()
            # ax.errorbar(x,y=np.mean(props_1,axis=0), yerr=std1,xerr=0,linestyle='none',color='black',capsize=2.0)

            # plt.gca().spines['top'].set_visible(False)
            # plt.gca().spines['right'].set_visible(False)
            plt.ylabel(ylabel,fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.savefig(os.path.join(figure_dir,fname),bbox_inches='tight')
    
    def plot_bar_counts_v3(self,runs,labels,ylabel,colors,col='class',values_to_exclude=None,filt=['to_keep','pass_cutoff'], map_group=None,figure_dir="comparative_figures/control",fname="compare_class_counts.png",normalize=True,figsize=(10,5)):
        counts_dict={}
        std_cols = []
        for i,runs_i in enumerate(runs):
            props_i,index_i=self._get_counts(runs_i,col,filt,map_group=map_group,values_to_exclude=values_to_exclude,normalize=normalize)
            print(props_i)
            props_i=np.array(props_i)
            counts_dict[labels[i]] = np.mean(props_i,axis=0)
            print(counts_dict)
            counts_dict[f'std{i}']=np.std(props_i,axis=0)
            std_cols.append(f'std{i}')
        output_df=pd.DataFrame(counts_dict,index=index_i)
        print(output_df)
        context = {'font.sans-serif':"Arial",'font.family':"sans-serif","font.size":14,"figure.dpi":200,"legend.fontsize":14,"figure.figsize":figsize}
        print(output_df)
        with plt.rc_context(context):
            if len(runs)==1:
                plt.bar(x=[0,1,2,3],height=output_df[labels].values.reshape(-1), color=list(colors),label=list(output_df.index),yerr=list(output_df['std0']),capsize=3.0,edgecolor='black')
                plt.gca().set_xticks(ticks=[0,1,2,3], labels=list(output_df.index))
            else:
                ax = output_df[labels].plot(kind='bar', color=colors,yerr=output_df[std_cols].values.T,capsize=2.0,error_kw=dict(ecolor='k'))
        plt.ylabel(ylabel,fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(os.path.join(figure_dir,fname),bbox_inches='tight')
        plt.show()
    
    def compare_across_runs(self,col, run_groups,group_names=None,filt=True, hue_order=None,figure_dir=os.path.join("comparative_figures","rabies_control"),fname="compare_runs.png"):
        arr=[]
        for j,runs in enumerate(run_groups):
            for i,run in enumerate(runs):
                adata = sc.read_h5ad(os.path.join("data",run,self.final_f))
                print(adata.obs['class'].unique())
                df = adata.obs.copy()
                if filt:
                    df = df[df['to_keep'] & df['pass_cutoff']]
                if group_names is None:
                    df['Trial']=run
                else:
                    df['Trial'] = group_names[j]
                df['replicate']=i
                arr.append(df)
        df_grouped = pd.concat(arr)
        cmap = plt.get_cmap('Set1')
        with plt.rc_context({"font.size":14}):
            # ax = sns.countplot(data=df_grouped,x='Trial',hue=col,palette={"Other":cmap(0),'RGC':cmap(2),'Amacrine':cmap(4)}) easier but no error bars
            s = df_grouped.groupby(['Trial','replicate'])['class'].value_counts()
            s= s.rename('counts')
            df_p = s.reset_index()
            sns.barplot(data=df_p,x='Trial',hue=col,y='counts',errorbar='sd',hue_order=hue_order)
            plt.legend(bbox_to_anchor=[1.0,1.0])
            plt.savefig(os.path.join(figure_dir,fname),bbox_inches='tight')
            plt.show()
        return df
            
        
        


    
    def plot_violin_expression_v2(self,runs_1,runs_2,label1,label2,ylabel, colors,col='class',values_to_exclude=None,filt=['to_keep','pass_cutoff'], map_group=None,figure_dir="comparative_figures/control",fname="compare_class_counts.png",normalize=True):
        return
        

    
    def plot_class_scatter(self,run):
        subdirectories = [f"{run}_rg{i}" for i in self.regions_lst]
        adata = sc.read_h5ad(os.path.join(self.data_dir,run,self.final_f))
        removal_areas_per_region = rc.BIPOLAR_ZONES
        a = ap.AggregatePipeline(subdirectories,removal_areas_per_region)
        a.merged_ann = adata
        filt = adata.obs['to_keep'] & adata.obs['pass_cutoff']

        a.plot_scatter(filt,invert_yaxis=False,invert_xaxis=True)
        df=a.merged_ann[filt].obs
        print(np.sum(df['group'].map(a.GROUP_MAP)=="RGC"))
        print(np.sum(df['group'].map(a.GROUP_MAP)=="Amacrine"))
        print(np.sum(df['group']== 'pc/Am+/RGC+'))
        print(np.sum(df['group']== 'pc/Am-/RGC-'))
    
    def all_cells_umap(self,run):
        subdirectories = [f"{run}_rg{i}" for i in self.regions_lst]
        a = ap.AggregatePipeline(subdirectories,self.removal_areas_per_region)
        a.plot_all_umap(cluster_key='leiden',group_key='group_num',fname=self.reassigned_cells_f)
    
    def get_filtered_statistics(self,run,s=0.01,markerscale=10,bbox_to_anchor=[1.0,1.0]):
        subdirectories = [f"{run}_rg{i}" for i in self.regions_lst]
        a = ap.AggregatePipeline(subdirectories,self.removal_areas_per_region)
        a.get_filtered_statistics(s=s,markerscale=markerscale,bbox_to_anchor=bbox_to_anchor)
    
    def get_region_visualizer(self,run,region=0,downsample_ratio=4):
        subdirectory = f"{run}_rg{region}"
        stitched_m = np.load(os.path.join(self.data_dir,subdirectory, self.stitched_m_f))
        micron_regions = rc.RN_REGIONS[subdirectory]
        v= visualizer.Visualizer(micron_regions,stitched_m, downsample_ratio,self.im_script_dir,subdirectory)
        return v

    
    def get_summary_statistics(self,runs=None,name_map=None,fname=None):
        data=[]
        cluster_names = [f"C{i}" for i in range(1,46)]
        if runs is None:
            runs=self.runs.copy()
        for run in runs:
            summary_dict = {}
            f = os.path.join(self.data_dir,run, self.final_f)
            adata = sc.read_h5ad(f)
            filt = adata.obs['to_keep'] & adata.obs['pass_cutoff']

            summary_dict['Total # segmented cells'] = adata.X.shape[0]
            summary_dict['# cells filtered out manually removed bipolar cell regions'] = adata[~adata.obs['to_keep']].X.shape[0]
            summary_dict['# additional cells filtered out for low transcripts'] = adata[adata.obs['to_keep'] & ~adata.obs['pass_cutoff']].X.shape[0]
            summary_dict['% cells filtered'] = adata[~filt].X.shape[0]/summary_dict['Total # segmented cells']*100

            # df contains cells only used for analysis
            df = adata[filt].obs
            summary_dict['Total cells kept for analysis'] = adata[filt].X.shape[0]
            summary_dict['# Amacrine'] = df[df['final_assignment']=="Amacrine"].shape[0] 
            summary_dict['% Amacrine'] = summary_dict['# Amacrine']/df.shape[0]*100
            summary_dict['# Other'] = df[df['final_assignment']=='Other'].shape[0]
            summary_dict['% Other'] = summary_dict['# Other']/df.shape[0]*100
            n_class_RGC = df[df['final_assignment'].isin(cluster_names)].shape[0]
            n_unclass_RGC = df[df['final_assignment']=='Unassigned'].shape[0]
            summary_dict['# RGC'] = n_class_RGC+n_unclass_RGC
            summary_dict['% RGC'] = summary_dict['# RGC']/(df.shape[0])*100

            summary_dict['# Classified RGC'] = n_class_RGC 
            summary_dict['# Unclassified RGC'] =n_unclass_RGC
            summary_dict['% of RGCs classified'] = n_class_RGC/(n_class_RGC+n_unclass_RGC)*100
            data.append(summary_dict)
        if name_map is None:
            index=runs
        else:
            index = map(lambda x: name_map[x], runs)
        summary_df = pd.DataFrame(data=data,index=index)
        summary_df=summary_df.round(decimals=1)
        if fname is None:
            fname="summary_statistics.csv"
        summary_df.to_csv(os.path.join(self.subfolder,fname))
        return summary_df


        


        
        


        



    
