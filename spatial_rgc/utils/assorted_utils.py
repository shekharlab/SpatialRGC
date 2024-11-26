import time
import re
from random import sample

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.utils import shuffle
from anndata import AnnData
import xgboost as xgb
from matplotlib import gridspec
from matplotlib.pyplot import figure
import pandas as pd
import scanpy as sc
import os
import glob as glob
import seaborn as sns
from collections import OrderedDict

import anndata as ad

def plotConfusionMatrixNew(
    ytrue,
    ypred,
    type,
    xaxislabel,
    yaxislabel,
    title,
    train_dict,
    test_dict=None,
    re_order=None,
    re_order_cols = None,
    re_index = None,
    re_order_rows = None,
    save_as=None,
    c='blue',
    ):
    # Note: Train classes = actual classes + unassigned
    # ytrue = independent clustering (test), ypred = classifier prediction (train)
    # Class labels must be nonnegative consecutive integers ranging from 0,1,...,n_classes-1


    numbertrainclasses = len(train_dict)
    numbertestclasses = len(test_dict)
    confusion_matrix = np.zeros((numbertestclasses,numbertrainclasses))
    for i,_ in enumerate(ytrue):
        confusion_matrix[ytrue[i],ypred[i]] +=1

    # Normalize confusion matrix
    for row,arr in enumerate(confusion_matrix):
        row_sum = np.sum(arr)
        if row_sum !=0:
            confusion_matrix[row] = confusion_matrix[row]/row_sum

    conf_df = pd.DataFrame(confusion_matrix)

    conf_df.index = list(test_dict.keys()) 
    conf_df.columns = list(train_dict.keys())

    # Reorder rows to try and make diagonal
    if re_index:
        inv_dict = {v:k for k,v in test_dict.items()}
        most_likely = np.argmax(confusion_matrix, axis=0)
        row_order = [inv_dict[i] for i in list(set(most_likely))]
        #row_order = list(dict.fromkeys(most_likely)) # Note: If one type is most likely for multiple rows, this will get rid of the duplicates
        unclear_assignment = set(test_dict.keys()) - set(row_order)
        row_order.extend(unclear_assignment)
        #row_order = [inv_dict[i] for i in row_order]
        conf_df = conf_df.reindex(row_order)
        print(conf_df)
    diagcm = conf_df.to_numpy()

    if re_order and re_order_cols is None:
        col_assign = dict(zip(train_dict.values(),[0]*len(train_dict)))
        col_assign[train_dict['Unassigned']] = 1
        confusion_matrix= conf_df.values
        cutoff=0.05
        inv_dict = {v:k for k,v in train_dict.items()}
        re_order_cols=[]
        for i,row in enumerate(confusion_matrix):
            possibly_shift = np.where(row>cutoff)[0]
            for j in possibly_shift:
                if confusion_matrix[i][j] > col_assign[j]:
                    col_assign[j] = confusion_matrix[i][j]
                    re_order_cols.append(inv_dict[j])
        re_order_cols = list(reversed(re_order_cols))
        re_order_cols = list(OrderedDict((x, True) for x in re_order_cols).keys())
        re_order_cols = list(reversed(re_order_cols))
    elif re_order:        
        conf_df=conf_df[re_order_cols]

        
    
    #conf_df.index = conf_df.index.map(test_dict)
    xticksactual = conf_df.columns

    dot_max = np.max(diagcm.flatten())
    dot_min = 0
    if dot_min != 0 or dot_max != 1:
        frac = np.clip(diagcm, dot_min, dot_max)
        old_range = dot_max - dot_min
        frac = (frac - dot_min) / old_range
    else:
        frac = diagcm
    xvalues = []
    yvalues = []
    sizes = []
    for i in range(diagcm.shape[0]):
        for j in range(diagcm.shape[1]):
            xvalues.append(j)
            yvalues.append(i)
            sizes.append((frac[i,j]*35)**1.5)
    size_legend_width = 0.5
    height = diagcm.shape[0] * 0.3 + 1
    height = max([1.5, height])
    heatmap_width = diagcm.shape[1] * 0.35
    width = (
        heatmap_width
        + size_legend_width
        )
    fig = plt.figure(figsize=(width, height))
    axs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        wspace=0.02,
        hspace=0.04,
        width_ratios=[
                    heatmap_width,
                    size_legend_width
                    ],
        height_ratios = [0.5, 10]
        )
    dot_ax = fig.add_subplot(axs[1, 0])
    dot_ax.scatter(xvalues,yvalues, s = sizes, c = c, norm=None, edgecolor='none')
    y_ticks = range(diagcm.shape[0])
    dot_ax.set_yticks(y_ticks)
    if type == 'validation':
        dot_ax.set_yticklabels(list(train_dict.keys()))
    elif type == 'mapping':
        # dot_ax.set_yticklabels(list(test_dict.keys()))
        dot_ax.set_yticklabels(list(conf_df.index))
    
    #xticksactual = conf_df.columns.map({v:k for k,v in train_dict.items()})
    dot_ax.set_xticklabels(xticksactual,rotation=90)
    x_ticks = range(diagcm.shape[1])
    dot_ax.set_xticks(x_ticks)
    #dot_ax.set_xticklabels(xticksactual, rotation=90)
    dot_ax.tick_params(axis='both', labelsize='small')
    dot_ax.grid(True, linewidth = 0.2)
    dot_ax.set_axisbelow(True)
    dot_ax.set_xlim(-0.5, diagcm.shape[1] + 0.5)
    ymin, ymax = dot_ax.get_ylim()
    dot_ax.set_ylim(ymax + 0.5, ymin - 0.5)
    dot_ax.set_xlim(-1, diagcm.shape[1])
    dot_ax.set_xlabel(xaxislabel)
    dot_ax.set_ylabel(yaxislabel)
    dot_ax.set_title(title)
    size_legend_height = min(1.75, height)
    wspace = 10.5 / width
    axs3 = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=axs[1, 1],
        wspace=wspace,
        height_ratios=[
                    size_legend_height / height,
                    (height - size_legend_height) / height
                    ]
        )
    diff = dot_max - dot_min
    if 0.3 < diff <= 0.6:
        step = 0.1
    elif diff <= 0.3:
        step = 0.05
    else:
        step = 0.2
    fracs_legends = np.arange(dot_max, dot_min, step * -1)[::-1]
    if dot_min != 0 or dot_max != 1:
        fracs_values = (fracs_legends - dot_min) / old_range
    else:
        fracs_values = fracs_legends
    size = (fracs_values * 35) ** 1.5
    size_legend = fig.add_subplot(axs3[0])
    size_legend.scatter(np.repeat(0, len(size)), range(len(size)), s=size, c = c)
    size_legend.set_yticks(range(len(size)))
    labels = ["{:.0%}".format(x) for x in fracs_legends]
    if dot_max < 1:
        labels[-1] = ">" + labels[-1]
    size_legend.set_yticklabels(labels)
    size_legend.set_yticklabels(["{:.0%}".format(x) for x in fracs_legends])
    size_legend.tick_params(axis='y', left=False, labelleft=False, labelright=True)
    size_legend.tick_params(axis='x', bottom=False, labelbottom=False)
    size_legend.spines['right'].set_visible(False)
    size_legend.spines['top'].set_visible(False)
    size_legend.spines['left'].set_visible(False)
    size_legend.spines['bottom'].set_visible(False)
    size_legend.grid(False)
    ymin, ymax = size_legend.get_ylim()
    size_legend.set_ylim(ymin, ymax + 0.5)
    if save_as is not None:
        fig.savefig(save_as, bbox_inches = 'tight')
    plt.show()
    return diagcm, None,axs

def plot_mapping_new(test_labels, test_predlabels, test_dict, train_dict, xaxislabel, yaxislabel,re_order=None,re_order_cols = None,re_index = None, re_order_rows = None, save_as=None,c='blue'):
    
    ARI = adjusted_rand_score(labels_true = test_labels, 
                              labels_pred = test_predlabels)
    accuracy = np.sum(test_labels==test_predlabels)/len(test_labels)*100
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    with plt.rc_context({"figure.dpi":200,"font.sans-serif":'Arial'}):
        
        mappingconfmat, mappingxticks, mappingplot = plotConfusionMatrixNew(
        ytrue = test_labels,
        ypred = test_predlabels,
        test_dict=test_dict,
        train_dict=train_dict,
        type = 'mapping',
        save_as = save_as,
        title = 'ARI = {:.3f}, Accuracy = {:.1f}%'.format(ARI,accuracy),
        xaxislabel =xaxislabel,
        yaxislabel = yaxislabel,
            re_order=re_order,
        re_order_cols = re_order_cols,
            re_index = re_index,
        re_order_rows = re_order_rows,
        c=c,
        )    

def plot_mapping(test_labels, test_predlabels, test_dict, train_dict, 
                 xaxislabel, yaxislabel,
                re_order=None,
    re_order_cols = None,
                 re_index = None,
    re_order_rows = None, save_as=None,
    row_prefix='M'):
    
    ARI = adjusted_rand_score(labels_true = test_labels, 
                              labels_pred = test_predlabels)
    
           
    mappingconfmat, mappingxticks, mappingplot = plotConfusionMatrix(
    ytrue = test_labels,
    ypred = test_predlabels,
    test_dict=test_dict,
    train_dict=train_dict,
    type = 'not mapping',
    save_as = save_as,
    title = 'ARI = {:.3f}'.format(ARI),
    xaxislabel =xaxislabel,
    yaxislabel = yaxislabel,
        re_order=re_order,
    re_order_cols = re_order_cols,
        re_index = re_index,
    re_order_rows = re_order_rows,
    pre=row_prefix
    )    

def plotConfusionMatrix(
    ytrue,
    ypred,
    type,
    xaxislabel,
    yaxislabel,
    title,
    train_dict,
    test_dict=None,
    re_order=None,
    re_order_cols = None,
    re_index = None,
    re_order_rows = None,
    save_as=None,
    pre='M'
    ):
    #very bad
    numbertrainclasses = len(train_dict)
    # numbertestclasses = len(set(ytrue))
    
    confusionmatrix = confusion_matrix(y_true = ytrue, y_pred = ypred)

    #only need this when mapping b/c if validaiton, all classes will be used and cfm will be constructed properly
    if type == 'mapping':
        print(type)
        # ytrue = independent clustering, ypred = classifier prediction

        
        numbertestclasses = len(test_dict)
        # TEST
        num_labels = np.max((numbertrainclasses+1,numbertestclasses))
        confmatpercent = confusion_matrix(y_true=ytrue,y_pred=ypred,normalize='true',labels=np.arange(num_labels))
        confmatpercent = confmatpercent[0:numbertestclasses,:]
        print(confmatpercent)
        conf_df = pd.DataFrame(confmatpercent)
        conf_df.index = list(test_dict.keys())
       #ENDTEST
        #name columns of conf mat
        if(len(conf_df.columns)>len(train_dict)):
            conf_df.columns = list(train_dict.keys())+['Unassigned']
        else:
            conf_df.columns = list(train_dict.keys())


        if (re_order):
            conf_df = conf_df[re_order_cols]

        if (re_index):
            most_likely = np.argmax(confmatpercent, axis=0)
            row_order = list(dict.fromkeys(most_likely))
            unclear_assignment = set(range(len(test_dict)))-set(most_likely)
            row_order.extend(unclear_assignment)
            row_order = [pre+ str(x) for x in row_order]
            conf_df = conf_df.reindex(row_order)
        diagcm = conf_df.to_numpy()
        """
        """
    
        xticksactual = list(conf_df.columns)
    else:
        if numbertrainclasses in ypred:
            confusionmatrix = confusionmatrix[0:numbertrainclasses,0:numbertrainclasses+1]#for Unassigned
        else:
            confusionmatrix = confusionmatrix[0:numbertrainclasses,0:numbertrainclasses]
        confmatpercent = np.zeros(confusionmatrix.shape)
        for i in range(confusionmatrix.shape[0]):
            if np.sum(confusionmatrix[i,:]) != 0:
                confmatpercent[i,:] = confusionmatrix[i,:]/np.sum(confusionmatrix[i,:])
            else:
                confmatpercent[i,:] = confusionmatrix[i,:]
            diagcm = confmatpercent
            xticks = np.linspace(0, confmatpercent.shape[1]-1, confmatpercent.shape[1], dtype = int)
        xticksactual = []
        for i in xticks:
            if i != numbertrainclasses:
                xticksactual.append(list(train_dict.keys())[i])
            else:
                xticksactual.append('Unassigned')
    dot_max = np.max(diagcm.flatten())
    dot_min = 0
    if dot_min != 0 or dot_max != 1:
        frac = np.clip(diagcm, dot_min, dot_max)
        old_range = dot_max - dot_min
        frac = (frac - dot_min) / old_range
    else:
        frac = diagcm
    xvalues = []
    yvalues = []
    sizes = []
    for i in range(diagcm.shape[0]):
        for j in range(diagcm.shape[1]):
            xvalues.append(j)
            yvalues.append(i)
            sizes.append((frac[i,j]*35)**1.5)
    size_legend_width = 0.5
    height = diagcm.shape[0] * 0.3 + 1
    height = max([1.5, height])
    heatmap_width = diagcm.shape[1] * 0.35
    width = (
        heatmap_width
        + size_legend_width
        )
    fig = plt.figure(figsize=(width, height))
    axs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        wspace=0.02,
        hspace=0.04,
        width_ratios=[
                    heatmap_width,
                    size_legend_width
                    ],
        height_ratios = [0.5, 10]
        )
    dot_ax = fig.add_subplot(axs[1, 0])
    dot_ax.scatter(xvalues,yvalues, s = sizes, c = 'blue', norm=None, edgecolor='none')
    y_ticks = range(diagcm.shape[0])
    dot_ax.set_yticks(y_ticks)
    if type == 'validation':
        dot_ax.set_yticklabels(list(train_dict.keys()))
    elif type == 'mapping':
        # dot_ax.set_yticklabels(list(test_dict.keys()))
        dot_ax.set_yticklabels(list(conf_df.index))
    x_ticks = range(diagcm.shape[1])
    dot_ax.set_xticks(x_ticks)
    dot_ax.set_xticklabels(xticksactual, rotation=90)
    dot_ax.tick_params(axis='both', labelsize='medium')
    dot_ax.grid(True, linewidth = 0.2)
    dot_ax.set_axisbelow(True)
    dot_ax.set_xlim(-0.5, diagcm.shape[1] + 0.5)
    ymin, ymax = dot_ax.get_ylim()
    dot_ax.set_ylim(ymax + 0.5, ymin - 0.5)
    dot_ax.set_xlim(-1, diagcm.shape[1])
    dot_ax.set_xlabel(xaxislabel,fontsize=20)
    dot_ax.set_ylabel(yaxislabel,fontsize=20)
    dot_ax.set_title(title)
    size_legend_height = min(1.75, height)
    wspace = 10.5 / width
    axs3 = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=axs[1, 1],
        wspace=wspace,
        height_ratios=[
                    size_legend_height / height,
                    (height - size_legend_height) / height
                    ]
        )
    diff = dot_max - dot_min
    if 0.3 < diff <= 0.6:
        step = 0.1
    elif diff <= 0.3:
        step = 0.05
    else:
        step = 0.2
    fracs_legends = np.arange(dot_max, dot_min, step * -1)[::-1]
    if dot_min != 0 or dot_max != 1:
        fracs_values = (fracs_legends - dot_min) / old_range
    else:
        fracs_values = fracs_legends
    size = (fracs_values * 35) ** 1.5
    size_legend = fig.add_subplot(axs3[0])
    size_legend.scatter(np.repeat(0, len(size)), range(len(size)), s=size, c = 'blue')
    size_legend.set_yticks(range(len(size)))
    labels = ["{:.0%}".format(x) for x in fracs_legends]
    if dot_max < 1:
        labels[-1] = ">" + labels[-1]
    size_legend.set_yticklabels(labels)
    size_legend.set_yticklabels(["{:.0%}".format(x) for x in fracs_legends])
    size_legend.tick_params(axis='y', left=False, labelleft=False, labelright=True)
    size_legend.tick_params(axis='x', bottom=False, labelbottom=False)
    size_legend.spines['right'].set_visible(False)
    size_legend.spines['top'].set_visible(False)
    size_legend.spines['left'].set_visible(False)
    size_legend.spines['bottom'].set_visible(False)
    size_legend.grid(False)
    ymin, ymax = size_legend.get_ylim()
    size_legend.set_ylim(ymin, ymax + 0.5)
    fig.savefig(save_as, bbox_inches = 'tight')
    plt.show()
    return diagcm, xticksactual, axs

def plot_mapping(test_labels, test_predlabels, test_dict, train_dict, 
                 xaxislabel, yaxislabel,
                re_order=None,
    re_order_cols = None,
                 re_index = None,
    re_order_rows = None, save_as=None,
    row_prefix='M'):
    
    ARI = adjusted_rand_score(labels_true = test_labels, 
                              labels_pred = test_predlabels)
    
    
 
    
           
    mappingconfmat, mappingxticks, mappingplot = plotConfusionMatrix(
    ytrue = test_labels,
    ypred = test_predlabels,
    test_dict=test_dict,
    train_dict=train_dict,
    type = 'mapping',
    save_as = save_as,
    title = 'ARI = {:.3f}'.format(ARI),
    xaxislabel =xaxislabel,
    yaxislabel = yaxislabel,
        re_order=re_order,
    re_order_cols = re_order_cols,
        re_index = re_index,
    re_order_rows = re_order_rows,
    pre=row_prefix
    )    
      

#This helper method uses xgboost to train classifiers.
def trainclassifier(train_anndata, common_top_genes, obs_id, train_dict, eta,
                    max_cells_per_ident, train_frac, min_cells_per_ident, run_validation,run_full,colsample_bytree=1,nround=200,max_depth=4):
    
    start_time = time.time()
    unique_classes = np.unique(train_anndata.obs[obs_id].values)
    numbertrainclasses = len(unique_classes)

    xgb_params_train = {
            'objective':'multi:softprob',
            'eval_metric':'mlogloss',
            'num_class':numbertrainclasses,
            'eta':eta,
            'max_depth':max_depth,
            'subsample': 0.6,
            'colsample_bytree':colsample_bytree,
            'n_jobs': 6}
    #Train XGBoost on 70% of training data and validate on the remaining data


    training_set_train_70 = []
    validation_set_train_70 = []
    training_label_train_70 = []
    validation_label_train_70 = []
    bst_model_train_70 =  None
    bst_model_full_train = None
    validation_pred_train_70 = None
    if run_validation:
        print("Running train/validation split")
        #loop thru classes to split for training and validation
        for i in unique_classes:

            #how many cells in a class
            cells_in_clust = train_anndata[train_anndata.obs[obs_id]==i,:].obs_names #cell names
            n = min(max_cells_per_ident,round(len(cells_in_clust)*train_frac))

            #sample 70% for training and rest for validation
            train_temp = np.random.choice(cells_in_clust,n,replace = False)
            validation_temp = np.setdiff1d(cells_in_clust, train_temp)

            #upsample small clusters
            if len(train_temp) < min_cells_per_ident:
                train_temp_bootstrap = np.random.choice(train_temp, size = min_cells_per_ident - int(len(train_temp)))
                train_temp = np.hstack([train_temp_bootstrap, train_temp])

            #store training and validation **names** of cells in vectors, which update for every class
            training_set_train_70 = np.hstack([training_set_train_70,train_temp])
            validation_set_train_70 = np.hstack([validation_set_train_70,validation_temp])

            #store training and validation **labels** of cells in vectors, which update for every class
            training_label_train_70 = np.hstack([training_label_train_70,np.repeat(train_dict[i],len(train_temp))])
            validation_label_train_70 = np.hstack([validation_label_train_70,np.repeat(train_dict[i],len(validation_temp))])

            #need train_dict b/c XGboost needs number as class labels, not words
            #this is only deconvulted later in plotting function

        #put data in XGB format
        X_train = train_anndata[training_set_train_70,common_top_genes].X
        train_matrix_train_70 = xgb.DMatrix(data = X_train, label = training_label_train_70, 
                                            feature_names = common_top_genes)

        X_valid = train_anndata[validation_set_train_70,common_top_genes].X
        validation_matrix_train_70 = xgb.DMatrix(data = X_valid, label = validation_label_train_70, 
                                                 feature_names = common_top_genes)

        del training_set_train_70, validation_set_train_70, training_label_train_70

        #Train on 70%
        bst_model_train_70 = xgb.train(
            params = xgb_params_train,
            dtrain = train_matrix_train_70,
            num_boost_round = nround)
        
        validation_pred_train_70 = bst_model_train_70.predict(data = validation_matrix_train_70)
    if run_full:
        print("Training on 100%")
        #Train on 100%
        #Train XGBoost on the full training data
        training_set_train_full = []
        training_label_train_full = []

        for i in unique_classes:
            train_temp = train_anndata.obs.index[train_anndata.obs[obs_id].values == i]
            if len(train_temp) < 100:
                train_temp_bootstrap = np.random.choice(train_temp, size = 100 - int(len(train_temp)))
                train_temp = np.hstack([train_temp_bootstrap, train_temp])
            
            #indices of cells in class
            training_set_train_full = np.hstack([training_set_train_full,train_temp])
            
            #labels of cells in class: [label*N_class] stacked onto previous classes
            training_label_train_full = np.hstack([training_label_train_full,np.repeat(train_dict[i],len(train_temp))])


        X_train_full = train_anndata[training_set_train_full,common_top_genes].X
        full_training_data = xgb.DMatrix(data = X_train_full, label = training_label_train_full, 
                                        feature_names = common_top_genes)

        del training_set_train_full, training_label_train_full

        bst_model_full_train = xgb.train(
            params = xgb_params_train,
            dtrain = full_training_data,
            num_boost_round = nround
            )

    
    
    print('trainclassifier() complete after', np.round(time.time() - start_time), 'seconds')
    
    #real labels of validation set, predicted labels, classifier.
    #recall these are all integers that are deconvulted later in plotting using the dicts
    return validation_label_train_70, validation_pred_train_70, bst_model_train_70, bst_model_full_train


#This helper method predicts the testing cluster labels.
def predict(train_anndata, common_top_genes, bst_model_train_full, test_anndata, 
            train_obs_id, test_dict, test_obs_id):
    
    #Predict the testing cluster labels
    #how many classes mapping to 
    numbertrainclasses = len(np.unique(train_anndata.obs[train_obs_id].values))
    
    #put testing data into XGB format
    full_testing_data = xgb.DMatrix(data = test_anndata[:,common_top_genes].X, 
                                    feature_names=common_top_genes)
    
    #a testing_cells x numclasses matrix, with each vector containing prob association with the classes
    test_prediction = bst_model_train_full.predict(data = full_testing_data)

    #for each cell, go through vec of probs and take index of max prob (if greater than ...): that's assignment

    
    test_predlabels = np.zeros((test_prediction.shape[0]))
    for i in range(test_prediction.shape[0]):
        if np.max(test_prediction[i, :]) > 1.1*(1/numbertrainclasses):
            test_predlabels[i] = np.argmax(test_prediction[i,:])
        
        #"unassigned" is a label one larger than all b/c python begins indexing at 0
        else:
            test_predlabels[i] = numbertrainclasses
        
    test_labels = np.zeros(len(test_anndata.obs[test_obs_id].values))
    for i,l in enumerate(test_anndata.obs[test_obs_id].values):
        test_labels[i] = test_dict[l]

    #actual labels of testing set, the labels that test set mapped to 
    return test_labels, test_predlabels, test_prediction

# Update to take names
def cutoff_predict(X, model, unassigned_indicator, cutoff,gene_names=None):
    # A MORE NUMPY VERSION OF THE ABOVE
    if gene_names is None:
        y_probs = model.predict(xgb.DMatrix(data = X))
    else:
        y_probs = model.predict(xgb.DMatrix(data = X,feature_names=gene_names))
    y_pred = threshold(y_probs,unassigned_indicator,cutoff)
    return y_pred,y_probs

def threshold(y_probs,unassigned_indicator,cutoff):
    maxes = np.max(y_probs,axis=1)
    max_indices = np.argmax(y_probs,axis=1)
    y_pred = np.where(maxes < cutoff, unassigned_indicator, max_indices)
    return y_pred

def overlapping_genes(gene_names_old, gene_names_new):
    for gene in gene_names_old:
        item = re.search(r'FAM19A4', gene)
        if item is not None:
            print(gene)
    for gene in gene_names_new:
        item = re.search(r'TAFA4', gene)
        if item is not None:
            print(gene)

def load_data_cellpose(amp, gene_matrix, processed_rgc):
    "Load new dataset and add a few things"
    adata_merfish = sc.read_h5ad(amp)
    print(amp)
    adata_merfish.obs = adata_merfish.obs.rename(columns={"col":"center_x", "row":"center_y"})
    adata_merfish.obs['barcodeCount'] = adata_merfish.X.sum(axis=1)
    adata_merfish.var['expression'] = adata_merfish.X.sum(axis=0)
    # Need this to get gene names in correct order
    # cell_by_gene = pd.read_csv(gene_matrix, index_col=0)
    print("HACK TO WORKAROUND NOT HAVING CELL_BY_GENE")
    cell_by_gene = pd.read_csv("imaging_scripts/mappings/140g_rn8_rg1/cell_by_gene.csv",index_col=0)

    gene_names_new = [x for x in cell_by_gene.columns if 'Blank' not in x]

    adata_merfish.obsm["spatial"] = adata_merfish.obs[['center_x', 'center_y']]
    "Old dataset"
    processed_rgc = sc.read_h5ad(processed_rgc)
    gene_names_old = processed_rgc.var.index.tolist()
    return processed_rgc, adata_merfish, gene_names_old, gene_names_new

# New functinos
def load_data(gene_matrix, metadata, processed_rgc):
    "Load new dataset and add a few things"
    cell_by_gene = pd.read_csv(gene_matrix, index_col=0)
    meta_cell = pd.read_csv(metadata, index_col=0)
    print("Total number of cells: {}".format(len(cell_by_gene)))
    cells = cell_by_gene.index.tolist()
    meta_cell['barcodeCount'] = cell_by_gene.sum(axis=1)
    meta_cell = meta_cell.loc[cells]
    meta_gene = pd.DataFrame(index=cell_by_gene.columns)
    keep_genes = [x for x in cell_by_gene.columns if 'Blank' not in x]
    cell_by_gene = cell_by_gene[keep_genes]
    meta_gene = meta_gene.loc[keep_genes]
    meta_gene['expression'] = cell_by_gene.sum(axis=0)
    adata_merfish = sc.AnnData(X=cell_by_gene.values, obs=meta_cell, var=meta_gene)
    adata_merfish.obsm["spatial"] = adata_merfish.obs[['center_x', 'center_y']]
    # JOSHgene_names_new =  adata_merfish.var.index.str.upper().tolist()
    gene_names_new = adata_merfish.var.index.tolist()
    "Old dataset"
    processed_rgc = sc.read_h5ad(processed_rgc)
    gene_names_old = processed_rgc.var.index.tolist()
    
    return processed_rgc,adata_merfish,gene_names_old,gene_names_new

def load_merfish(gene_matrix,metadata):
    cell_by_gene = pd.read_csv(gene_matrix, index_col=0)
    meta_cell = pd.read_csv(metadata, index_col=0)
    cells = cell_by_gene.index.tolist()
    meta_cell = meta_cell.loc[cells]
    meta_cell.index = range(len(meta_cell.index))
    meta_gene = pd.DataFrame(index=cell_by_gene.columns)
    keep_genes = [x for x in cell_by_gene.columns if 'Blank' not in x]
    cell_by_gene = cell_by_gene[keep_genes]
    meta_gene = meta_gene.loc[keep_genes]
    adata_merfish = sc.AnnData(X=cell_by_gene.values, obs=meta_cell, var=meta_gene)
    adata_merfish.obsm["spatial"] = adata_merfish.obs[['center_x', 'center_y']]
    # JOSHgene_names_new =  adata_merfish.var.index.str.upper().tolist()
    gene_names_new = adata_merfish.var.index.tolist()
    return adata_merfish,gene_names_new

def load_scrnaseq(h5ad_path):
    processed_rgc = sc.read_h5ad(h5ad_path)
    gene_names_old = processed_rgc.var.index.tolist()
    return processed_rgc,gene_names_old
    
    

"""Generate categorical labels: 0->C1, 1->C2, 2->C3..."""
def generate_labels(adata, key,new_key):
    clust_ids = sorted(adata.obs[key].unique())
    clust_names = ['C' + str(x+1) for x in clust_ids]
    train_dict = dict(zip(clust_names,clust_ids))
    cluster_names = []
    for value in adata.obs[key]:
        cluster_names.append('C' + str(value+1))
    adata.obs[new_key] = adata.obs[key].apply(lambda x: 'C' + str(x+1))
    return train_dict



def preprocess_old_data(old_adata, genes_to_use,key='annotated',map_names={'Fam19a4':'Tafa4','Tusc5':'Trarg1'}):
    """
    NOTE: By inspection, two genes in the new dataset {TRARG1, TAFA4} were not found in the old dataset. 
    Turns out these genes were called {TUSC5,FAM19A4} previously
    Thus we rename them in the old dataset
    """    
    if map_names is not None:
        old_adata.var.rename(index=map_names,inplace=True)
    old_adata = old_adata[:,genes_to_use]
    if key =='annotated':
        train_dict = generate_labels(old_adata, key=key,new_key='cluster_names')
    else:
        train_dict={}

    """More transform"""
    old_adata.X = old_adata.layers['raw'].todense().copy()
    # JOSHold_adata.layers['raw'] = old_adata.X.copy()
    sc.pp.normalize_total(old_adata, inplace=True)
    med_lib_size = np.round(np.sum(old_adata.X[0]))
    old_adata.layers['median_transform'] = old_adata.X.copy()
    sc.pp.log1p(old_adata,copy=False)
    old_adata.layers['log_transform'] = old_adata.X.copy()
    print(f"Median library size calculated from relevant genes: {med_lib_size}")
    return old_adata, train_dict, med_lib_size

def fill_genes(adata,genes):
        genes_to_add = set(genes) - set(adata.var.index.values)
        new_X = np.concatenate((adata.X, np.zeros((adata.X.shape[0],len(genes_to_add)))),axis=1)
        new_var = pd.DataFrame(index=list(adata.var.index.values) + list(genes_to_add))
        new_var.index.name='gene'
        print(new_var)
        return ad.AnnData(X=new_X,obs=adata.obs,var=new_var,obsm=adata.obsm)

def preprocess_new_data(adata_merfish, genes_to_use,med_lib_size,threshold_rm=10,threshold_am=5,r_merge=4): #rm=10, am=5 default
    "Process new dataset using median lib size from old one; set aside un log transformed data"
    # adata_new.var.index = adata_new.var.index.str.upper()
    if 'max_areas' in adata_merfish.obs.index:
        adata_merfish.obs['Soma radius'] = np.sqrt(adata_merfish.obs['max_areas']/np.pi)

    is_present = [x for x in adata_merfish.var.index.values if x in genes_to_use]
    #assert len(is_present) == len(adata_merfish.var.index.values), (is_present,adata_merfish.var.index.values)
    if set(adata_merfish.var.index.values) != set(genes_to_use):
        mer = fill_genes(adata_merfish,genes_to_use)[:,genes_to_use]
    else:
        mer = adata_merfish[:,genes_to_use]
    
    print("Number of cells prefiltering: {}".format(mer.X.shape[0]))
    gene_counts = mer.X.sum(axis=1)
    # max_counts = np.percentile(gene_counts, 90)
    min_counts = max(15, np.percentile(gene_counts, 10))
    sc.pp.filter_cells(mer, min_counts = min_counts,inplace=True)
    adata_merfish.obs['pass_cutoff'] = (np.sum(adata_merfish.X, axis=1) >= max(15, np.percentile(gene_counts, 10)))


    print("Number of cells with transcripts >15", np.sum(np.sum(mer.X,axis=1) > 15) )
    # RGC_markers = []
    """
    for marker in RGC_markers:
        plt.hist(mer[:,marker].X.copy().flatten(),range=(0,10),density=True)
        plt.title(marker)
        plt.show()
    """
    method = {0:'and',1:'or',2:'sum'}
    m_id = 2
    if m_id == 0:
        RGC_markers = {"Rbpms":8, "Pou4f1":1,"Pou4f2":1, "Slc17a6":1}
        for marker in RGC_markers:
            count_thresh = RGC_markers[marker]
            mer = mer[mer[: , marker].X >= count_thresh, :]        
    elif m_id == 1:
        marked = False
        RGC_markers = {"Rbpms":5, "Pou4f1":1,"Pou4f2":1, "Slc17a6":1}
        for marker in RGC_markers:
            count_thresh = RGC_markers[marker]
            marked = marked | (mer[:,marker].X >= count_thresh)
        mer = mer[marked,:]
    elif m_id==2:
        RGC_markers = ["Rbpms", "Pou4f1", "Pou4f2", "Slc17a6"]
        amacrine_markers = ["Tfap2a", "Tfap2b", "Chat"]
        # Filters cells with too few counts of RGC markers
        rm_counts = np.sum(mer[:,RGC_markers].X,axis=1)
        mer = mer[rm_counts> threshold_rm,:]
        adata_merfish.obs['is_RGC'] = np.array((np.sum(adata_merfish[:,RGC_markers].X, axis=1) > threshold_rm))
        # print(f"Number of cells remaining after filtering out non-RGCs: {mer.obs.shape[0]}")
        # Filter cells with too many counts of amacrine markers
        am_counts = np.sum(mer[:,amacrine_markers].X,axis=1)
        mer = mer[am_counts<threshold_am, :]
        adata_merfish.obs['is_amacrine'] = np.array((np.sum(adata_merfish[:,amacrine_markers].X, axis=1) >= threshold_am))
        # print(f"Number of cells remaining after filtering out amacrine: {mer.obs.shape[0]}")

    else:
        raise AssertionError("method not specified")
    vol_cutoff = 100
    size_cutoff = 50
    if 'volume' in mer.obs.columns:
        mer = mer[(mer.obs['volume'] >= vol_cutoff) , :]
    elif 'Size' in mer.obs.columns:
        mer = mer[(mer.obs['Size'] >= size_cutoff) , :]
    else:
        print("WARNING: Not filtering by size nor volume since neither key is present")
    # print("Filtering criteria: {}, {}, m_id: {}".format(RGC_markers, vol_cutoff, method[m_id]))
    # print("Min and max counts: {} {}".format(min_counts,max_counts))
    # print(f"Median library size used for new genes: {med_lib_size}")
    # sc.pp.filter_cells(mer, max_counts = max_counts,inplace=True)
    # Basic filter for most likely to be RGC
    print("Number of cells post-filtering: {}".format(mer.X.shape[0]))
    # mer.layers['raw'] = mer.X.copy()
    sc.pp.normalize_total(mer,target_sum=med_lib_size, inplace=True)
    # mer.layers['median_transform'] = mer.X.copy()
    sc.pp.log1p(mer,copy=False)
    # mer.layers['log_transform'] = mer.X.copy()
    return mer

def generate_models(train_data, gene_names, train_dict,cutoffs,run_validation,run_full, model_loc,fig_loc,post,model_name=None):
    if model_name is None:
        model_name = "xgb{}{}".format(cutoffs,post)
    unassigned_indicator = len(train_dict)
    y, y_probs, valid_model,model = trainclassifier(train_anndata=train_data, 
                                             common_top_genes=gene_names, 
                                             obs_id='cluster_names', 
                                             train_dict=train_dict, 
                                             eta=0.2,
                                             max_cells_per_ident=2000, train_frac=0.7,
                                             min_cells_per_ident=100,
                                             run_validation=run_validation,
                                             run_full=run_full)
    if run_validation:
        valid_model.save_model(os.path.join(model_loc,"valid_"+model_name))
        np.save(os.path.join(model_loc,"y_label{}".format(post)), y)
        np.save(os.path.join(model_loc,"y_probs{}".format(post)), y_probs)
        for cutoff in cutoffs:
            y_pred = threshold(y_probs,unassigned_indicator,cutoff)
            plot_validation_plots(y, y_pred, train_dict=train_dict, cutoff=cutoff,save=os.path.join(fig_loc,"Cutoff{}_{}.png".format(cutoff,post)))
    if run_full:
        model.save_model(os.path.join(model_loc,model_name))



def cutoffs_old_data_check(version, unassigned_indicator, model_loc, fig_loc,post,title):
    cutoff_fns = glob.glob(os.path.join(fig_loc,'Cutoff*.png'))
    accuracies = []
    ARIs = []
    unassigned = []
    cutoffs = set()
    for fn in cutoff_fns:
        srch = re.search('Cutoff(.*?)_(.*).png',fn)
        if srch is not None:
            cutoff_str = srch.group(1)
            if float(cutoff_str) % 1 == 0:
                cutoffs.add(int(cutoff_str))
            else:
                cutoffs.add(float(cutoff_str))
    cutoffs = list(cutoffs)
    cutoffs.sort()
    y = np.load(os.path.join(model_loc,'y_label{}.npy'.format(post)))
    y_probs = np.load(os.path.join(model_loc,'y_probs{}.npy'.format(post)))
    for cutoff in cutoffs:
        y_pred = threshold(y_probs, unassigned_indicator,cutoff)
        ARI = adjusted_rand_score(labels_true = y, 
                                  labels_pred = y_pred)
        accuracy = (y==y_pred).astype(int).sum()/len(y)
        frac_unassigned = (y_pred==unassigned_indicator).astype(int).sum()/len(y)
        accuracies.append(accuracy)
        ARIs.append(ARI)
        unassigned.append(frac_unassigned)
    plt.figure()
    cmap = mpl.cm.get_cmap('tab10')
    plt.plot(cutoffs, accuracies, linewidth=1.5,color=cmap(0), label="Accuracy")
    plt.plot(cutoffs, unassigned, linewidth=1.5, color=cmap(1),label="Fraction Unassigned")
    plt.plot(cutoffs, ARIs, linewidth=1.5,color=cmap(2), label="ARI")
    plt.xlabel('Probability cutoff')
    plt.ylabel('Metric')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(fig_loc, "Old_dataset_metrics_{}{}.png".format(version,post)))
    plt.show()
    return accuracies, ARIs, unassigned


def cutoffs_new_data(adata_new, unassigned_indicator, cutoffs, model_loc, figure_loc,post):
    unassigned= []
    model = xgb.Booster()
    model.load_model(os.path.join(model_loc,"xgb{}{}".format(cutoffs,post)))
    for cutoff in cutoffs:    
        y_pred,_ = cutoff_predict(X=adata_new.X, model=model, unassigned_indicator=unassigned_indicator, cutoff=cutoff)
        frac_unassigned = (y_pred==unassigned_indicator).astype(int).sum()/len(y_pred)
        unassigned.append(frac_unassigned)
    plt.figure() 
    plt.plot(cutoffs,unassigned)
    plt.xlabel("Probability cutoff")
    plt.ylabel("Fraction unassigned")
    plt.title("Merfish: Cells unassigned vs. probability cutoff")
    plt.savefig(os.path.join(figure_loc,"Unassigned_merfish_cells{}.png".format(cutoffs)))


"""# Get a sense of distribution of unassigned to assigned

NOTE: Below modifies adata_new_log in place"""
def plot_cluster_assignments(adata_new_log, train_dict, cutoffs, model_loc, figure_loc,post):
    model = xgb.Booster()
    model.load_model(os.path.join(model_loc,"xgb{}{}".format(cutoffs,post)))
    unassigned_indicator = len(train_dict)
    for cutoff in cutoffs:
        y_pred,_ = cutoff_predict(X=adata_new_log.X, model=model, 
                                              unassigned_indicator= unassigned_indicator, 
                                              cutoff=cutoff)
        clusters = []
        for y in y_pred:
            if y != unassigned_indicator:
                clusters.append("C" + str(y+1))
            else:
                clusters.append("Unassigned")
        adata_new_log.obs['cluster_names'] = clusters
        plot_tmp = adata_new_log.obs[adata_new_log.obs['cluster_names'] != "Unassigned"]
        figure(figsize=(30, 15), dpi=150)
        total_counts = np.sum(plot_tmp['cluster_names'].value_counts())
        ax = (plot_tmp['cluster_names'].value_counts()/total_counts).plot.bar()
        pct_assigned = len(plot_tmp)/len(adata_new_log.obs)*100
        print(pct_assigned/100 * len(adata_new_log.obs))
        ax.set_title(f"Frequency of clusters (Total cells:{len(adata_new_log.obs)}, % assigned: {pct_assigned:.1f}, cutoff={cutoff})")
        for p in ax.patches:
            ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
        plt.ylabel("Normalized frequency")
        plt.savefig(os.path.join(figure_loc, "Frequency of clusters, cutoff={}.png".format(cutoff)))
        plt.show()
    return adata_new_log

def plot_assignments(mer, model, train_dict, cutoffs, FIGURES,suffix=None):
    unassigned_indicator = len(train_dict)
    for cutoff in cutoffs:
        y_pred,_ = cutoff_predict(X=mer.X, model=model, 
                                              unassigned_indicator= unassigned_indicator, 
                                              cutoff=cutoff)
        clusters = []
        for y in y_pred:
            if y != unassigned_indicator:
                clusters.append("C" + str(y+1))
            else:
                clusters.append("Unassigned")
        mer.obs['cluster_names'] = clusters
        plot_tmp = mer.obs[mer.obs['cluster_names'] != "Unassigned"]
        figure(figsize=(30, 15), dpi=150)
        total_counts = np.sum(plot_tmp['cluster_names'].value_counts())
        ax = (plot_tmp['cluster_names'].value_counts()/total_counts).plot.bar()
        pct_assigned = len(plot_tmp)/len(mer.obs)*100
        print(pct_assigned/100 * len(mer.obs))
        ax.set_title(f"Frequency of clusters (Total cells:{len(mer.obs)}, % assigned: {pct_assigned:.1f}, cutoff={cutoff})")
        for p in ax.patches:
            ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
        plt.ylabel("Normalized frequency")
        plt.savefig(os.path.join(FIGURES, f"Frequency of clusters, cutoff={cutoff}.png_{suffix}.png"))
        plt.show()
    return mer

def compare_cluster_assignments(adata_new,adata_old,ck_n,ck_o, train_dict,plot,fname):
    unassigned_indicator = len(train_dict)
    p=np.zeros(len(train_dict))
    q=np.zeros(len(train_dict))
    inv_d = {v: k for k, v in train_dict.items()}
    
    m_c = adata_new.obs[adata_new.obs[ck_n] != unassigned_indicator][ck_n].astype(int)
    print(m_c)
    q = np.bincount(m_c)/len(m_c)
    p = np.bincount(adata_old.obs[ck_o])/len(adata_old.obs[ck_o])
    # bd_dist = bd_distance(p,q)
    cnames = [inv_d[x] for x in range(len(train_dict))]
    if plot:
        plt.clf()
        figure(figsize=(20, 4.8), dpi=200) # 'lightsteelblue', 'lightcoral'
        """
        plt.bar(cnames,q,width=0.7,align='center',label='MERFISH')
        plt.bar(cnames,p,width=0.2,align='center',label='scRNA-seq')
        """
        x=np.arange(len(train_dict))
        width=0.3
        cmap = mpl.cm.get_cmap('Pastel1')
        print(cmap)
        plt.bar(x,q,width=width,align='center', edgecolor="black", color=cmap(0),label='MERFISH')
        plt.bar(x+width,p,width=width,align='center', edgecolor="black", color=cmap(1), label='scRNA-seq')
        plt.xticks(x+width/2,cnames)
        plt.legend()
        plt.xlabel("RGC Type")
        plt.ylabel("Normalized Frequency")
        # plt.title(f"Bhattacharyya distance: {bd_dist}")
        plt.minorticks_off()
        plt.xlim([-1,45])
        plt.tick_params(axis='x')
        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('left')
        plt.tick_params(axis='x',rotation=90)
        plt.savefig(fname,bbox_inches='tight')
        plt.show()
    return

def bd_distance(p,q):
    return -np.log(np.sum(np.sqrt(p*q)))

        

"""Pick a subset of n genes for MERFISH"""

"""Approach 1: Fix n1 genes and select n-n_1 genes based off log2FC and % expression"""
def recommend_genes(df, n_genes,c_key,fc_key, in_key,out_key,gene_key, min_fc,max_fc, clust_thresh, other_thresh, order, preselected_genes=None):
    VALID_ORDERINGS = [c_key, fc_key]
    if order not in VALID_ORDERINGS:
        raise ValueError(
            "Invalid choice of ordering '{}' was chosen. Please choose between {}".format(order,VALID_ORDERINGS))
    mask = (df[fc_key] >= min_fc) & (df[fc_key] <= max_fc) & (df[in_key] >= clust_thresh) & (df[out_key] <= other_thresh)
    df_filt = df[mask]
    print
    print("Number of genes that satisfy criteria: {}".format(len(df_filt[gene_key].unique())))
    print("Number of clusters that have a gene: {}".format(len(df_filt[c_key].unique())))    
    gene_intersection = set(df_filt[gene_key]).intersection(set(preselected_genes))
    print("Number of genes intersecting with first MERFISH selection: {}".format(len(gene_intersection)))
    return df_filt[gene_key], gene_intersection
    # df_filt[c_key].hist()
    # df[c_key].hist()
    # print(len(np.unique(df[gene_key])),len(df[gene_key]))


def recommend_genes_v2(df, n_genes,c_key,fc_key, in_key,out_key,gene_key, min_fc,max_fc, clust_thresh, other_thresh, order, preselected_genes=None,ascending=False):
    VALID_ORDERINGS = ['pct.1','pct.2', fc_key]
    if order not in VALID_ORDERINGS:
        raise ValueError(
            "Invalid choice of ordering '{}' was chosen. Please choose between {}".format(order,VALID_ORDERINGS))
    select_genes = set()
    if preselected_genes is not None:
        select_genes = select_genes.union(preselected_genes)
    n2 = n_genes - len(select_genes)
    terminate = False
    df2 = df.groupby(c_key).apply(lambda x: x.sort_values(by=order,ascending=False))
    i=0
    c_names = sorted(np.unique(df[c_key]))
    while not terminate:
        for c_name in c_names:
            if len(select_genes) >= n2:
                terminate = True
                break
            df3 = df2[df2[c_key]==c_name][gene_key]
            if len(df3) > i:
                gene = df2[df2[c_key] == c_name][gene_key].iloc[i]
                if gene not in select_genes:
                    select_genes.add(gene)
                else:
                    print("dup genes")
                    # print(c_name, gene,i)
        i+=1
    print(len(select_genes))
    print(select_genes)
    gene_per_clust = de_df[de_df[gene_key].isin(select_genes)][c_key].value_counts()
    print(gene_per_clust)
    return select_genes,gene_per_clust

def recommend_genes_v3(df, n_genes,c_key,diff_key, in_key,out_key,gene_key, min_fc,max_fc, clust_thresh, other_thresh, order, preselected_genes=None,ascending=False):
    VALID_ORDERINGS = ['diff', 'pct_nz_reference']
    if order not in VALID_ORDERINGS:
        raise ValueError(
            "Invalid choice of ordering '{}' was chosen. Please choose between {}".format(order,VALID_ORDERINGS))
    select_genes = []
    """
    if preselected_genes is not None:
        select_genes = select_genes.union(preselected_genes)
    """
    n2 = n_genes - len(select_genes)
    terminate = False
    df2 = df.groupby(c_key).apply(lambda x: x.sort_values(by=order,ascending=False))
    c_names = sorted(np.unique(df[c_key]))
    indices_map = dict(zip(c_names,np.zeros(len(c_names),dtype='int')))
    print("Creating list")
    while not terminate:
        for c_name in c_names:
            if len(select_genes) >= n2:
                terminate = True
                break
            df3 = df2[df2[c_key]==c_name][gene_key]
            looping = True
            while looping and indices_map[c_name]<len(df3):
                gene = df2[df2[c_key] == c_name][gene_key].iloc[indices_map[c_name]]
                if gene not in select_genes:
                    select_genes.append(gene)
                    # print(c_name,gene)
                    looping=False
                indices_map[c_name] += 1
    # get max diffs for each gene and associate with a cluster
    max_diffs = dict()
    cg_df = dict()
    cg_df['cluster'] = c_names
    for gene in select_genes:
        cg_df[gene] = np.zeros(len(c_names))
    cg_df = pd.DataFrame(cg_df)
    for gene in select_genes:
        df2 = df[df[gene_key] == gene].sort_values(by=diff_key, ascending=False)
        if len(df2) == 0:
            print("Gene {} not differentially expressed".format(gene))
        else:
            max_diff = df2.iloc[0][diff_key]
            max_diffs[max_diff] = gene 
            subset_cnames = df2[c_key].unique()
            cg_df.loc[cg_df['cluster'].isin(subset_cnames), gene] = 1
    # gene_per_clust = de_df[de_df[gene_key].isin(select_genes)][c_key].value_counts()
    # print(gene_per_clust)
    cg_df['annotated'] = cg_df['cluster'].apply(lambda x: int(x[1:])-1)
    cg_df = cg_df.sort_values(by='annotated',ascending=True)
    return select_genes,cg_df, max_diffs