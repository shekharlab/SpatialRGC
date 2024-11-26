import os
import numpy as np
import pandas as pd
import xgboost as xgb
import spatial_rgc.utils.assorted_utils as utils
import pickle as pickle

def train_model(adata,obs_id, train_dict, model_dir,model_name, cutoff, common_top_genes=None, eta=0.2,max_cells_per_ident=2000,train_frac=0.7,min_cells_per_ident=100,run_validation=True,run_full=False,unassigned_key="Unassigned",colsample_bytree=1,nround=200,max_depth=4):
    if common_top_genes is None:
        common_top_genes= list(adata.var.index)
    y, y_probs, valid_model,model=utils.trainclassifier(train_anndata=adata, 
                                                        common_top_genes=common_top_genes, 
                                                        obs_id=obs_id, # Clusters
                                                        train_dict=train_dict,  #Cluster: value
                                                        eta=eta,
                                                        max_cells_per_ident=max_cells_per_ident, train_frac=train_frac,
                                                        min_cells_per_ident=min_cells_per_ident,
                                                        run_validation=run_validation,
                                                        run_full=run_full,
                                                        colsample_bytree=colsample_bytree,
                                                        nround=nround,
                                                        max_depth=max_depth
                                                        )
    if model is not None:
        model.save_model(os.path.join(model_dir,model_name))
    if valid_model is not None:
        valid_model.save_model(os.path.join(model_dir,f"valid_{model_name}"))

    y_pred = utils.threshold(y_probs,unassigned_indicator=train_dict[unassigned_key],cutoff=cutoff)                                            
    np.save(os.path.join(model_dir,f"{model_name}_valid_y"),y)
    np.save(os.path.join(model_dir,f"{model_name}_valid_y_pred"),y_pred)
    np.save(os.path.join(model_dir,f"{model_name}_valid_y_probs"),y_probs)
    if model is not None:
        model.save_model(os.path.join(model_dir,model_name))
    if valid_model is not None:
        valid_model.save_model(os.path.join(model_dir,f"valid_{model_name}"))
    return y,y_pred,valid_model,model


def gen_dict(adata,col,prefix,unassigned_indicator):
    classes = adata.obs[col].unique().categories

    if all([i.isnumeric() for i in classes]):
        classes = sorted(classes.astype(int))

    if prefix is not None:
        classes = [f'{prefix}{i}' for i in classes]

    n_classes = len(classes)
    t_dict= dict(zip(classes, range(n_classes)))

    if unassigned_indicator is not None:
        t_dict[unassigned_indicator]=n_classes

    return t_dict
    
    




"""Vignette"""
def run_training_and_validation(adata,obs_id, model_dir,model_name,output_dir,output_f, train_dict_name, train_dict=None,run_validation=True,run_full=False,xlabel='Predicted',ylabel='True',
                cutoff=0.8,eta=0.2,max_cells_per_ident=2000,train_frac=0.7,min_cells_per_ident=100, colsample_bytree=1,nround=200, max_depth=4, unassigned_indicator='Unassigned'):
    if train_dict is None:
        train_dict = gen_dict(adata,obs_id,prefix='',unassigned_indicator=unassigned_indicator)

    print(train_dict)
    assert isinstance(train_dict,dict)
    with open(os.path.join(model_dir,train_dict_name),'wb') as f:
        pickle.dump(train_dict,f)
    y,y_pred,valid_model,model = train_model(
        adata,
        obs_id, 
        train_dict, 
        model_dir,
        model_name, 
        cutoff, 
        common_top_genes=None, 
        eta=eta,
        max_cells_per_ident = max_cells_per_ident,
        train_frac=train_frac,
        min_cells_per_ident = min_cells_per_ident,
        run_validation = run_validation,
        run_full = run_full,
        unassigned_key=unassigned_indicator,
        colsample_bytree=colsample_bytree,
        nround=nround,
        max_depth=max_depth
    )
    #test_dict = gen_dict(adata,obs_id,prefix=None,unassigned_indicator=None)
    #print(train_dict,test_dict)
    test_dict = {k:v for k,v in train_dict.items() if k != unassigned_indicator}
    utils.plot_mapping_new(test_labels=y.astype(int), #y-axis
                            test_predlabels=y_pred,#x-AXIS
                            test_dict=test_dict,  # y-axis
                            train_dict=train_dict, # x-axis
                            re_order=True,
                            re_order_cols=None,
                            re_index=True,
                            xaxislabel=xlabel, yaxislabel=ylabel,
                            save_as=os.path.join(output_dir,output_f)
    )
    return valid_model,model,train_dict,None


def apply_model(adata,model_f,train_dict,cutoff, output_dir,output_f, test_dict=None,unassigned_indicator='Unassigned', prefix='MC', test_key='leiden',train_key_added='retina_class',prefix_key=None,reorder_cols=None,xlabel='Predicted',ylabel='Clustering'): #Test key is usually clustering
    if isinstance(train_dict,str):
        with open(train_dict,'rb') as f:
            train_dict = pickle.load(f)
    if isinstance(model_f, str):
        model=xgb.Booster()
        model.load_model(model_f)
    else:
        model=model_f # Otherwise directly provide model

    assert isinstance(train_dict,dict)
    if test_dict is None:
        test_dict = gen_dict(adata,col=test_key,prefix=prefix,unassigned_indicator=None)
    """
    if prefix_key is None:
        prefix_key = f"{prefix}_{test_key}"
    """


    y_pred,y_probs = utils.cutoff_predict(X=adata.X, model=model, unassigned_indicator=train_dict[unassigned_indicator], cutoff=cutoff)
    test_labels = adata.obs[test_key]
    if adata.obs[test_key][0].isnumeric():
        test_labels=adata.obs[test_key].astype(int)
    else:
        test_labels = adata.obs[test_key].map(test_dict)

    adata.obs[train_key_added] = pd.Series(y_pred).map({v:k for k,v in train_dict.items()}).values
    print(test_labels.unique(),np.unique(y_pred))
    print(test_dict,train_dict)
    import importlib
    importlib.reload(utils)
    print(len(test_dict),len(train_dict))

    utils.plot_mapping_new(test_labels=test_labels, #y-axis
                            test_predlabels=y_pred,#x-AXIS
                            test_dict=test_dict,  # y-axis
                            train_dict=train_dict, # x-axis
                            re_order=True,
                            re_order_cols=None,
                            re_index=True,
                            xaxislabel=xlabel, yaxislabel=ylabel,
                            save_as=os.path.join(output_dir,output_f)
                )
    return test_dict






