import os
BASE_DIR = os.path.join("..","..","spatial_rgc")
INPUT_TRAIN_DICT_F = os.path.join(BASE_DIR,"models","removed_genes","train_dict.pickle")
XGB_F=os.path.join(BASE_DIR,"models","150g","xgb_150g")
# XGB_F_132=os.path.join(BASE_DIR,"models","removed_genes","xgb_132g")
XGB_F_130=os.path.join(BASE_DIR,"models","removed_genes","xgb_rgc130")
XGB_F_RETINA_CLASS=os.path.join(BASE_DIR,"models","retina_class_level","valid_xgb_retina_merged")
XGB_F_RETINA_CLASS=os.path.join(BASE_DIR,"models","retina_class_level","valid_xgb_retina_merged")
# XGB_F_RETINA_MACOSKO=os.path.join(BASE_DIR,"models","retina_class_level","valid_xgb_retina_macosko")
# XGB_F_RETINA_SAC=os.path.join(BASE_DIR,"models","retina_class_level","valid_xgb_retina_class_SAC")
# XGB_F_RETINA_SAC_F=os.path.join(BASE_DIR,"models","retina_class_level","xgb_retina_class_SAC")




MODEL_INFO={
        "tran_yan_ONC_class":{
            "data_f":os.path.join(BASE_DIR,"data","aggregated","cell_class_merged_atlas.h5ad"),'median_transcripts':174, "feature_names_f": os.path.join(BASE_DIR,"models","removed_genes","feature_names_130.pickle"),"train_dict_name":"train_dict_merged.pickle",
            "model_dir":os.path.join(BASE_DIR,"models","retina_class_level"),"model_name":"xgb_retina_merged","out":"merged",'feature_names_upper_f':os.path.join(BASE_DIR,"models","removed_genes","feature_names_130.pickle")
        },
    }

#RGC_MEDIAN_TRANSCRIPTS_DICT = {XGB_F:176, XGB_F_132:175,XGB_F_130:174,XGB_F_RETINA_CLASS:174, XGB_F_RETINA_MACOSKO:174,XGB_F_RETINA_SAC:174,XGB_F_RETINA_SAC_F:174}
RGC_MEDIAN_TRANSCRIPTS_DICT = {XGB_F:176,XGB_F_130:174,XGB_F_RETINA_CLASS:174}

FEATURE_NAMES_F_130 = os.path.join(BASE_DIR,"models","removed_genes","feature_names_130.pickle")



MERGED_F="Cellpose_merged.h5ad"
MERGED_REMAPPED_F="Cellpose_merged_remapped.h5ad",
ALL_REASSIGNED_CELLS_F="all_cells_reassigned.h5ad"
RGC_cache_f="RGC_only.h5ad"

NUM_RGC_TYPES = 45
GROUP_MAP={'pc/Am+/RGC+': "Other", 'pc/Am+/RGC-':"Amacrine", 'pc/Am-/RGC+':"RGC", 'pc/Am-/RGC-':"Other",'dpc/Am+/RGC+': "N/A", 'dpc/Am+/RGC-':"N/A", 'dpc/Am-/RGC+':"N/A", 'dpc/Am-/RGC-':"N/A"}
RENAMED_GENE_MAP = {'Fam19a4':'Tafa4','Tusc5':'Trarg1'} # keys are atlas gene names, values are merfish gene names

id_to_cluster={i:f'C{i+1}' for i in range(45)}
id_to_cluster[45]='Unassigned'

orientation={
    "140g_rn3":{'invert_xaxis':True,'invert_yaxis':False}, 
    "140g_rn4":{'invert_xaxis':True, 'invert_yaxis':True},
    "140g_rn5": {'invert_xaxis':True,'invert_yaxis':True},
    #"140g_rn8": {'invert_xaxis':False,'invert_yaxis':True}
    }

BAE_CELL_INFO = "cell_info.mat"
BAE_GC_LIST = "gc_list.mat"
#Tran to Bae type mapping

tran_bae_mapping = {
    'C33': '1ws',#M1a 
    'C40':'1ws_duplicate', #M1b, duplicate is not actual type, just given unique id for purpose of plotting
    'C45':'4ow', #alpha-off-T
    'C41':'6sw', #alpha-on-T
    'C43':'8w', #alpha-onS/M4
    'C42':'1wt',#alpha-off-S
    'C3':'63', #Fmini on
    'C4':'2an', #Fmini off,
    'C28':'2aw', #Fmidi off
    'C38': '6t', #Fmidi on
    'C31': '9w', #M2
    'C6': '51', #W3b/LED
}

AMACRINE_MAP={
    'GabaAC':[1,2,5,6,7,8,11,14,17,18,20,21,22,23,25,26,27,29,31,32,34,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,57,58,59,60,61,63],
    'GlyAC':[3,4,9,12,13,15,19,28,33,35],
    'Neither':[10,24,30,36],
    'Both':[16,53,62]
}


#SC_GENES = ['Adam12', 'Adarb2', 'Ankfn1', 'Arpp21', 'Atp5md', 'Barhl1', 'C1ql2', 'Cacna2d1', 'Cadps2', 'Calb1', 'Calb2', 'Car10', 'Cbln2', 'Cbln4', 'Cdh13', 'Cdh18', 'Cdh4', 'Cdh7', 'Cdh8', 'Cdhr1', 'Chrm2', 'Chrm3', 'Col25a1', 'Cplx3', 'Cpne4', 'Csgalnact1', 'Cxcl14', 'Dmbx1', 'Ebf1', 'Ebf3', 'Epha3', 'Erbb4', 'Etv1', 'Fabp5', 'Fibcd1', 'Fign', 'Fos', 'Foxp2', 'Fras1', 'Fstl4', 'Gabrr2', 'Gad1', 'Gad2', 'Galnt14', 'Gap43', 'Gata3', 'Gda', 'Gfra1', 'Gpc3', 'Gpc5', 'Hcn1', 'Hdac9', 'Hs3st4', 'Htr2c', 'Il1rapl2', 'Junb', 'Kcnab1', 'Kcnd3', 'Kcng3', 'Kcng4', 'Kctd8', 'Kirrel3', 'Lhfp', 'Lhx1os', 'Lmo1', 'Lmo3', 'Maf', 'Marcksl1', 'Megf11', 'Meis1', 'Meis2', 'Mgarp', 'Mgat4c', 'Mllt11', 'Myh8', 'Nap1l5', 'Npas3', 'Npas4', 'Npffr1', 'Npnt', 'Nrtn', 'Ntng1', 'Ntng2', 'Nxph1', 'Otx2', 'Pax7', 'Pcbp3', 'Pcdh20', 'Pde1c', 'Pde5a', 'Pdzd2', 'Pmfbp1', 'Pou4f2', 'Ppia', 'Prkd1', 'Ptprk', 'Ptprt', 'Rasgrf2', 'Rd3l', 'Reln', 'Rnf152', 'Robo1', 'Rorb', 'Satb1', 'Sdk2', 'Sema3e', 'Sgcd', 'Slc17a6', 'Slc19a1', 'Smim17', 'Snap25', 'Sntb1', 'Sorcs1', 'Sorcs3', 'Sox14', 'Sox6', 'Sphkap', 'Spon1', 'Stmn1', 'Sv2c', 'Syt1', 'Syt17', 'Syt2', 'Syt4', 'Tac1', 'Tacr1', 'Tal1', 'Tcf7l2', 'Tfap2b', 'Tmem132c', 'Tmem132d', 'Tnni3', 'Tnnt1', 'Tpd52l1', 'Trim66', 'Trpm3', 'Tshz2', 'Vwc2', 'Zfhx3', 'Zfhx4']