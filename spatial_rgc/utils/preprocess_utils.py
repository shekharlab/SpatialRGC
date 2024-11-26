import pickle as pickle
import scanpy as sc
import xgboost as xgb
import numpy as np


import spatial_rgc.utils.assorted_utils as utils

def preprocess(adata,target_sum=None):
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
    print("Target sum:",np.median(np.sum(adata.X,axis=1)))
    sc.pp.log1p(adata, copy=False)
    sc.pp.scale(adata,copy=False)

def rearrange_and_preprocess(adata,feature_names_f,target_sum=None, batch_key=None,use_rep='X',save_log=True):
    """Rearranges var.index order to match XGBoost model training set, and transforms data
    """
    if isinstance(feature_names_f,list):
        feature_names=feature_names_f
    elif isinstance(feature_names_f,str):
        with open(feature_names_f,"rb") as f:
            feature_names = pickle.load(f)
    else:
        raise TypeError("Feature names is not a string or list")
    try:
        del adata.uns['log1p']
        del adata.layers['log1p']
    except:
        "Log1p not done yet"
    adata= adata[:,feature_names].copy()
    adata.X = adata.layers['raw'].copy()
    if save_log:
        adata.layers['log1p'] = np.log1p(adata.layers['raw'])
    preprocess(adata,target_sum=target_sum)
    if batch_key is not None:
        sc.external.pp.bbknn(adata,batch_key=batch_key,use_rep=use_rep)
    return adata
    
def cluster(adata,n_neighbors=30,resolution=0.8,key_added='leiden',use_rep='X',maxiter=None):
    """Clusters in-place
    Args:
        adata_f (anndata): Should already be preprocessed
    Modifies:
        Updates adata_f in-place with neighborhood graph and clustering
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep,copy=False)
    sc.tl.umap(adata,copy=False,maxiter=None)
    sc.tl.leiden(adata, resolution=resolution,key_added=key_added,copy=False)

def classify(adata, class_col,num_classes,unassigned_indicator,xgb_model_f,cutoff=0.5, train_dict=None):
    """Classifies (RGC) types in placed
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

def generate_RGC_classifications(adata, feature_names_f, num_classes, xgb_model_f, unassigned_indicator=None, target_sum=None,cutoff=0.5,batch_key=None,n_neighbors=30,resolution=1.0,class_col='joint_type',key_added='leiden',train_dict=None):
    if unassigned_indicator is None:
        unassigned_indicator = num_classes

    adata= rearrange_and_preprocess(adata,target_sum=target_sum,feature_names_f=feature_names_f,batch_key=batch_key)
    cluster(adata,n_neighbors=n_neighbors,resolution=resolution,key_added=key_added)
    train_dict = classify(adata,class_col=class_col,num_classes=num_classes,unassigned_indicator=unassigned_indicator,xgb_model_f=xgb_model_f,cutoff=cutoff,train_dict=train_dict)
    return adata

