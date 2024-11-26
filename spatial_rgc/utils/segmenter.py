from cellpose import models
from cellpose.io import imread
from cellpose import io,utils,plot

import ClusterMap.clustermap as cm
import numpy as np

import cv2

import os
import subprocess
import pandas as pd



class Segmenter():
    SUPPORTED_MODELS={
        "cellpose":{'cyto2','nuclei','pretrained'},
        "baysor":{'transcripts only','prior_segmentation'},
        "clustermap":{'no_dapi',"with_dapi"}
    }

    def __init__(self, name, models_dir=os.path.join("imaging_scripts","models"),cache_dir=os.path.join("imaging_scripts","segmenter_cache")):
        self.name=name
        self.models_dir=models_dir
        self.cache_dir=cache_dir
    
        
    def _assign_transcripts(self,tdf,masks,cache=True):
        if len(masks.shape)==2:
            pixel_coords = tdf[["p_y","p_x"]].values
        elif len(masks.shape) ==3:
            pixel_coords = tdf[["global_z","p_y","p_x"]].values
        else:
            assert AssertionError(f"Segmented mask is neither 2D nor 3D and has shape {masks.shape} instead")
        tdf["cell_id"] = masks[tuple(pixel_coords.T)]
        if cache:
            output_f = os.path.join(self.cache_dir,f"{self.name}_transcripts.csv")
            tdf.write_csv(output_f,index=False)


    def run_segmentation(self,img, model_group, subtype, tdf=None, use_gpu=False,**kwargs):
        assert model_group in self.SUPPORTED_MODELS
        assert subtype in self.SUPPORTED_MODELS[model_group]
        print(kwargs)
        if model_group == "cellpose":
            masks = self._run_cellpose(img, subtype, args=kwargs, use_gpu=use_gpu)
        elif model_group == "baysor":
            args=kwargs.copy()
            args['subtype']=subtype
            args['transcripts']=tdf
            masks= self._run_baysor(**args)
        elif model_group == "clustermap":
            args=kwargs.copy()
            args['img']=img
            args['subtype']=subtype
            args['tdf']=tdf
            masks=self._run_clustermap(**args)
        return masks

    """
    kwargs should include optional arguments for function eval here: https://cellpose.readthedocs.io/en/latest/api.html
    """
    def _run_cellpose(self,img, subtype,args, use_gpu=False):
        model=None
        if subtype=='pretrained':
            assert 'model_name' in args
            model_p = os.path.join(self.models_dir,args['model_name'])
            model = models.CellposeModel(pretrained_model=model_p,gpu=use_gpu)
            print(f"Using finetuned Cellpose model stored at {model_p}")
        else:
            model = models.Cellpose(model_type=subtype,gpu=use_gpu)
            print(f"Using pretrained Cellpose model {subtype}")
        
        args = args.copy()
        args['x']=img
        masks,_,_,_ = model.eval(**args)# masks flows styles, diams
        return masks
    

    def _run_baysor(self,subtype, do_3D=False, x_col='global_x',y_col='global_y',z_col='global_z',gene_col='gene',min_transcripts=15,prior_conf=0.2,scale=11,cell_types=4, transcripts=None,id_col='cell_id'):     
        if isinstance(transcripts, str):
            transcripts_f=transcripts
        elif isinstance(transcripts, pd.DataFrame):
            transcripts_f =os.path.join(self.cache_dir,f"{self.name}_raw.csv")
            transcripts.to_csv(transcripts_f)
        cmd = ["./julia-1.8.3/bin/baysor", "run",transcripts_f]
        if subtype=="prior_segmentation":
            cmd.append(f":{id_col}")
            cmd.append(f'--prior-segmentation-confidence={prior_conf}')    
        cmd.extend([f"-s{scale}",f"-x{x_col}", f'-y{y_col}', f'-g{gene_col}', f'-m{min_transcripts}',f'--n-clusters={cell_types}'])
        cmd.extend([f"-o{os.path.join(self.cache_dir,self.name)}","--count-matrix-format=tsv","-p"])
        if do_3D:
            cmd.append(f'-z {z_col}')
        print(cmd)
        subprocess.run(cmd)

        


    def _run_clustermap(self,img,subtype,tdf, xy_radius=25, x_col='p_x',y_col='p_y',z_col='global_z',gene_col='gene', z_radius=0, num_dims=2,contamination=0.1,pct_filter=0.3):
        if subtype=="no_dapi":
            tdf_s = tdf.rename(columns={x_col:'spot_location_1',y_col:'spot_location_2',z_col:'spot_location_3',gene_col:'gene_name'})
            print(tdf_s)
            tdf_s['spot_location_3']=tdf_s['spot_location_3'].astype(int)
            genes = np.unique(tdf_s['gene_name'])
            gene_list = list(range(1,len(genes)+1))
            tdf_s['gene'] = tdf_s['gene_name'].map(dict(zip(genes,gene_list)))
            tdf_s.dropna(inplace=True)
            tdf_s.index = list(range(tdf_s.shape[0]))
            tdf_s['gene']=tdf_s['gene'].astype(int)

            model = cm.ClusterMap(spots=tdf_s, dapi=img[:,:,0].copy(), gene_list=gene_list, num_dims=num_dims, xy_radius=xy_radius, z_radius=z_radius,fast_preprocess=False,gauss_blur=False,sigma=1)
            model.min_spot_per_cell=20
            model.preprocess(dapi_grid_interval=25, LOF=True, contamination=contamination, pct_filter=pct_filter)
            model.segmentation(cell_num_threshold=0.1,add_dapi=True)
            model.plot_segmentation(plot_with_dapi=True,plot_dapi=True)
        else:
            raise AssertionError("Not implemented")










    

