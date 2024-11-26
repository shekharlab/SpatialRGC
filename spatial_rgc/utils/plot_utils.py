import importlib
import os
import re
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import scanpy as sc
import scipy
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics.cluster import adjusted_rand_score

import spatial_rgc.utils.aligner as aligner
import spatial_rgc.utils.rconstants as rc




# Helpful plots

"""
def plot_size_dist(mer, title,FIGURES):
    ordering = ['C'+ str(x) for x in range(1,46)]
    ordering.append('Unassigned')

    mer.obs['cluster_names'] = pd.Categorical(mer.obs['cluster_names'],categories=ordering)
    mer.obs['Soma volume'] = 4/3*np.pi*mer.obs['Soma radius']**3
    def boxplot_sorted(df, by, column):
        df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
        meds = df2.median().sort_values(ascending=True)
        df2[meds.index].boxplot(showfliers=False, vert=False,figsize=(8,20))
        plt.grid(visible=None)
        plt.xlim(0,2750)
        plt.xlabel(r'Soma Volume ($\mu m^3$)',fontsize=fontsize)
        plt.ylabel(r'RGC Type',fontsize=fontsize)
        plt.gca().yaxis.set_label_coords(-0.14, 0.5)
        plt.tick_params(axis='both',labelsize=fontsize)
        plt.title(title)
        plt.savefig(os.path.join(FIGURES, "soma size distribution.png"),dpi=200,bbox_inches='tight')

    fontsize=24
    boxplot_sorted(mer.obs,by='cluster_names',column='Soma volume')
    plt.show()
    # mer.obs.boxplot(column="Soma volume",by="cluster_names", figsize=(8,20), showfliers=False, flierprops={'markersize':2},vert=False,positions=mer.obs.index.values)
"""

def plot_cell_centroids(mer,title,FIGURES):
    plt.scatter(mer.obs['center_x'],mer.obs['center_y'],s=0.2)
    plt.xlabel(r'x ($\mu$m)')
    plt.ylabel(r'y ($\mu$m)')
    plt.savefig(os.path.join(FIGURES, "RGC_cell_centroid_coordinates.png"))
    plt.title(title)
    plt.show()

def plot_classification_threshold(mer, title, FIGURES):
    # Compare cell type size distributions
    fontsize=24
    ordering = ['C'+ str(x) for x in range(1,46)]
    y_probs = mer.obsm['y_probs']
    mer.obs['cluster_probs'] = np.max(y_probs, axis=1)
    mer.obs['cluster_no_threshold'] = (np.argmax(y_probs,axis=1) + 1).astype(str)
    mer.obs['cluster_no_threshold'] = mer.obs['cluster_no_threshold'].apply(lambda x: 'C' + x)
    mer.obs['cluster_no_threshold'] = pd.Categorical(mer.obs['cluster_no_threshold'],categories=ordering)
    ordering = ['C'+ str(x) for x in range(1,46)]
    ordering.append('Unassigned')
    mer.obs.boxplot(column="cluster_probs",by="cluster_no_threshold",figsize=(20,6),showfliers=False,flierprops={'markersize':2})
    plt.plot(np.arange(46) ,[0.5]*46,color='r',label='Model threshold',linestyle='dashed',alpha=0.7)
    plt.suptitle('')
    plt.ylabel(r'Maximum probability assignment',fontsize=fontsize)
    plt.xlabel(r'RGC Type',fontsize=fontsize)
    plt.title('')
    plt.grid(visible=None)
    plt.tick_params(axis='x',labelsize=fontsize,rotation=90)
    plt.tick_params(axis='y',labelsize=fontsize)
    plt.xlim(0,46)
    plt.minorticks_off()
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(FIGURES,"Model probability comparisons.png"),dpi=200,facecolor='w',bbox_inches='tight')
    plt.show()

def plot_cluster_frequency_bars(adata_1,adata_2,ck_1,ck_2, label_1,label_2,train_dict,unassigned_indicator,figure_dir, fname):
    p=np.zeros(len(train_dict)-1)
    q=np.zeros(len(train_dict)-1)
    inv_d = {v: k for k, v in train_dict.items()}
    
    m_c = adata_1.obs[adata_1.obs[ck_1] != unassigned_indicator][ck_1].astype(int)
    o_c = adata_2.obs[adata_2.obs[ck_2] != unassigned_indicator][ck_2].astype(int)

    q = np.bincount(m_c,minlength=len(train_dict)-1)/len(m_c)
    p = np.bincount(o_c,minlength=len(train_dict)-1)/len(o_c)
    # bd_dist = bd_distance(p,q)
    cnames = [inv_d[x] for x in range(len(train_dict)-1)]
    print(cnames)
    plt.clf()
    plt.figure(figsize=(20, 4.8), dpi=200) # 'lightsteelblue', 'lightcoral'
    """
    plt.bar(cnames,q,width=0.7,align='center',label='MERFISH')
    plt.bar(cnames,p,width=0.2,align='center',label='scRNA-seq')
    """
    x=np.arange(len(train_dict)-1)
    width=0.3
    cmap = mpl.cm.get_cmap('Pastel1')
    print(cmap)
    plt.bar(x,q,width=width,align='center', edgecolor="black", color=cmap(0),label=label_1)
    plt.bar(x+width,p,width=width,align='center', edgecolor="black", color=cmap(1), label=label_2)
    plt.xticks(x+width/2,cnames)
    plt.legend()
    plt.xlabel("RGC Type")
    plt.ylabel("Normalized Frequency")
    # plt.title(f"Bhattacharyya distance: {bd_dist}")
    plt.minorticks_off()
    plt.xlim([-1,unassigned_indicator])
    plt.tick_params(axis='x')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.tick_params(axis='x',rotation=90)
    plt.savefig(os.path.join(figure_dir,fname),bbox_inches='tight')
    plt.show()
    return


def plot_type_dist_scatter(adata_x,adata_y,title,xlabel,ylabel, type_col_x='annotated',type_col_y='cluster',figure_dir="comparative_figures",fname="type_distribution_scatter.png",unassigned_indicator=45,color='blue'):
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams.update({'font.size': 24})
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(8,8),dpi=200)
    tmp_y = adata_y[adata_y.obs[type_col_y] != unassigned_indicator] 
    tmp_x = adata_x[adata_x.obs[type_col_x] != unassigned_indicator]
    # unassigned_indicator = num_calsses
    y_seq_prop = np.bincount(tmp_y.obs[type_col_y],minlength=unassigned_indicator)/len(tmp_y.obs[type_col_y])*100
    x_seq_prop = np.bincount(tmp_x.obs[type_col_x],minlength=unassigned_indicator)/len(tmp_x.obs[type_col_x])*100
    print(len(x_seq_prop))
    print(len(y_seq_prop))
    ax.scatter(x_seq_prop,y_seq_prop,s=30,color=color)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.title(title)

    max_x=9
    corr_matrix = np.corrcoef(x_seq_prop,y_seq_prop)
    r = corr_matrix[0,1]
    plt.text(5, 7, 'r = %0.2f' % r,fontsize=18)

    ax.set_xlim(0,max_x)
    ax.set_ylim(0,max_x)
    x_y_l = np.linspace(0,9,num=50)
    ax.plot(x_y_l,x_y_l,linewidth=1,linestyle='dotted')
    fig.savefig(fname=f'{figure_dir}/{fname}',bbox_inches="tight")
    fig.show()

def plot_type_dist_scatter_v2(adata_arr_x,ann_arr_y,title,xlabel,ylabel, type_col_x='annotated',type_col_y='cluster',figure_dir="comparative_figures",fname="type_distribution_scatter.png",unassigned_indicator=45):
    context = {'font.sans-serif':"Arial",'font.family':"sans-serif","font.size":24}
    
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(8,8),dpi=200)
    tmp_y = adata_y[adata_y.obs[type_col_y] != unassigned_indicator] 
    tmp_x = adata_x[adata_x.obs[type_col_x] != unassigned_indicator]
    # unassigned_indicator = num_calsses
    y_seq_prop = np.bincount(tmp_y.obs[type_col_y],minlength=unassigned_indicator)/len(tmp_y.obs[type_col_y])*100
    x_seq_prop = np.bincount(tmp_x.obs[type_col_x],minlength=unassigned_indicator)/len(tmp_x.obs[type_col_x])*100
    print(len(x_seq_prop))
    print(len(y_seq_prop))
    ax.scatter(x_seq_prop,y_seq_prop,s=30)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.title(title)

    max_x=9
    corr_matrix = np.corrcoef(x_seq_prop,y_seq_prop)
    r = corr_matrix[0,1]
    plt.text(5, 7, 'r = %0.2f' % r,fontsize=18)

    ax.set_xlim(0,max_x)
    ax.set_ylim(0,max_x)
    x_y_l = np.linspace(0,9,num=50)
    ax.plot(x_y_l,x_y_l,linewidth=1,linestyle='dotted')
    fig.savefig(fname=f'{figure_dir}/{fname}',bbox_inches="tight")
    fig.show()


def boxplot_sorted(df, by, column,xlabel,ylabel,figure_dir,fname,xlim=None,values_to_exclude=None,showfliers=False):
    if values_to_exclude is not None:
        df = df[~df[by].isin(values_to_exclude)]
        df[by] = df[by].cat.remove_unused_categories()
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    meds = df2.median().sort_values(ascending=True)
    with plt.rc_context({"figure.figsize":[8,20], "figure.dpi":300, "font.sans-serif":'Arial',"font.family":"sans-serif","font.size":24}):
        df2[meds.index].boxplot(showfliers=showfliers, vert=False,figsize=(8,20))
        plt.grid(visible=None)
        if xlim is not None:
            plt.xlim(xlim)
        fontsize=24
        plt.xlabel(xlabel,fontsize=fontsize)
        plt.ylabel(ylabel,fontsize=fontsize)
        # plt.gca().yaxis.set_label_coords(-0.16, 0.5)
        # plt.tick_params(axis='both',labelsize=fontsize)
        plt.savefig(os.path.join(figure_dir, fname),dpi=200,bbox_inches='tight')

def stripplot_sorted(df, by, column,xlabel,ylabel,figure_dir,fname,xlim=None,values_to_exclude=None,showfliers=False):
    if values_to_exclude is not None:
        df = df[~df[by].isin(values_to_exclude)]
        df[by] = df[by].cat.remove_unused_categories()
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    meds = df2.median().sort_values(ascending=False)
    with plt.rc_context({"figure.figsize":[8,20], "figure.dpi":300, "font.sans-serif":'Arial',"font.family":"sans-serif","font.size":24}):
        sns.set_style("ticks")
        sns.stripplot(data=df,x=column,y=by, order=meds.index,orient="h")
        if xlim is not None:
            plt.xlim(xlim)
        fontsize=24
        plt.xlabel(xlabel,fontsize=fontsize)
        plt.ylabel(ylabel,fontsize=fontsize)
        # plt.gca().yaxis.set_label_coords(-0.16, 0.5)
        # plt.tick_params(axis='both',labelsize=fontsize)
        plt.savefig(os.path.join(figure_dir, fname),dpi=200,bbox_inches='tight')
    return meds
        

def plot_sizes_dist(adata,by,area_col,xlabel,ylabel,xlim=(0,3000), values_to_exclude=None, figure_dir="comparative_figures/control",fname="soma_size_distribution.png"):
    radii = np.sqrt(adata.obs[area_col]/np.pi)
    adata.obs['Volume'] = 4/3*np.pi*(radii)**3
    df = adata.obs 
    boxplot_sorted(df,by=by,column='Volume',xlabel=xlabel,ylabel=ylabel,figure_dir=figure_dir,fname=fname,xlim=xlim,values_to_exclude=values_to_exclude)

# Uses seaborn instead
def plot_sizes_dist_v2(adata, by, area_col,xlabel,ylabel,xlim=(0,3000),values_to_exclude=None,figure_dir="comparative_figures/control",fname="soma_size_distribution.png"):
    df=adata.obs
    if values_to_exclude is not None:
        df= df[~df[by].isin(values_to_exclude)]
    df['Soma Volume']=4/3*np.pi*(np.sqrt(df[area_col]/(np.pi)))**3
    order = df.groupby(by=by)['Soma Volume'].median().sort_values(ascending=False).index
    with plt.rc_context({"figure.figsize":[8,20], "figure.dpi":300, "font.sans-serif":'Arial',"font.family":"sans-serif","font.size":24}):
        sns.boxplot(data=df,x='Soma Volume', y=by,order=order,fliersize=0)
        plt.xlim(xlim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(figure_dir,fname),bbox_inches='tight')



def plot_gene_expression(adata_x,adata_y,xlabel,ylabel, var_names,categories, typename_col_x='cluster_names',typename_col_y='cluster_names',figure_dir="comparative_figures",fname="gene_expression_comparison",cmap='BuPu'):
    fig,ax = plt.subplots(nrows=2,figsize=(20,10))
    with plt.rc_context({"figure.dpi":200, "font.sans-serif":'Arial',"font.family":"sans-serif","font.size":14}):
        sc.pl.DotPlot(adata_x,var_names=var_names,categories_order=categories,groupby=typename_col_x, use_raw= False, cmap=cmap, title=xlabel,vmin=0, vmax=2,ax=ax[0],figsize=(30,6),DEFAULT_PLOT_X_PADDING=2).swap_axes().savefig('')
        sc.pl.DotPlot(adata_y, var_names=var_names,categories_order=categories, groupby=typename_col_y, use_raw=False,cmap=cmap, title=ylabel, vmin=0, vmax=2,ax=ax[1],figsize=(30,6),DEFAULT_PLOT_X_PADDING=2).swap_axes().savefig('')
        # fig.suptitle(f'{group}')
        plt.savefig(os.path.join(figure_dir,fname),dpi=200)
        plt.show()

def plot_gene_expression_v2(adata_x,adata_y,xlabel,ylabel, var_names,categories, typename_col_x='cluster_names',typename_col_y='cluster_names',figure_dir="comparative_figures",fname="gene_expression_comparison",cmap='BuPu'):
    with plt.rc_context({"figure.dpi":200, "font.sans-serif":'Arial',"font.family":"sans-serif","font.size":14}):
        sc.pl.DotPlot(adata_x,var_names=var_names,categories_order=categories,groupby=typename_col_x, use_raw= False, cmap=cmap, title=xlabel,vmin=0, vmax=2,figsize=(20,6)).swap_axes().savefig('')
        plt.savefig(os.path.join(figure_dir,f"{fname}_x.png"),dpi=200)
        plt.show()
        sc.pl.DotPlot(adata_y, var_names=var_names,categories_order=categories, groupby=typename_col_y, use_raw=False,cmap=cmap, title=ylabel, vmin=0, vmax=2,figsize=(20,6)).swap_axes().savefig('')
        plt.savefig(os.path.join(figure_dir,f"{fname}_y.png"),dpi=200)
        plt.show()

def plot_gene_expression_scatter(atlas,by_a,mer,by_m,lim=2, s=0.1, color='blue', figure_dir="control",fname="test.png"):
    classes = list(set(np.unique(atlas.obs[by_a])).intersection(set(np.unique(mer.obs[by_m]))))
    genes = list(set(atlas.var.index).intersection(set(np.unique(mer.var.index))))

    # Ensures order is correct
    print(genes)
    atlas = atlas[:,genes]
    mer=mer[:,genes]

    exp_arr_a=[]
    exp_arr_m=[]

    for c in classes:
        atlas_c = atlas[atlas.obs[by_a]==c]
        mer_c = mer[mer.obs[by_m]==c]
        exp_arr_a.extend(np.mean(atlas_c.X,axis=0))
        exp_arr_m.extend(np.mean(mer_c.X,axis=0))
    context = {'figure.figsize':(10,10),'font.sans-serif':"Arial",'font.family':"sans-serif","font.size":24,"figure.dpi":200}
    with plt.rc_context(context):
        plt.scatter(exp_arr_a,exp_arr_m,s=s,color=color)
        lim = lim
        plt.xlim(-lim,lim)
        plt.ylim(-lim,lim)
        plt.xlabel('Z-score expression (scRNA-seq)')
        plt.ylabel('Z-score expression (MERFISH)')
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        r=scipy.stats.pearsonr(exp_arr_a,exp_arr_m)
        plt.axline((0,0),slope=1,linestyle='--',linewidth=1,color='black')
        plt.text(x=0,y=-0.5,s="r={:.2f}".format(r.statistic))
        plt.savefig(os.path.join(figure_dir,fname),bbox_inches='tight')
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='(RGC type, gene) pair',
                          markerfacecolor=color, markersize=5)]
        plt.legend(handles=legend_elements,loc='upper right')
        plt.show()
        
"""
def cutoffs_test(adata,by,stain_col,min_cutoff,max_cutoff,group1,group2,figure_dir,fname, num=50,values_to_exclude=None,class_map=None,context=None,title=None):
    cutoffs = np.linspace(min_cutoff,max_cutoff,num)
    g1=[]
    g2=[]
    df = adata.obs[~adata.obs[by].isin(values_to_exclude)].copy()
    if class_map is not None:
            df['pseudo_group'] = df[by].map(class_map)
            by='pseudo_group'
    for cutoff in cutoffs:
        df['pass_cutoff'] = df[stain_col]>cutoff
        fracs_marked= df.groupby(by=by)['pass_cutoff'].sum()/(df[by].value_counts())
        g1.append(fracs_marked[group1])
        g2.append(fracs_marked[group2])
        print(f'{fracs_marked[group1]:.2f}',f'{fracs_marked[group2]:.2f}',cutoff)
    if context is None:
        context = {'figure.figsize':(8,8),'font.sans-serif':"Arial",'font.family':"sans-serif","font.size":24}
    with plt.rc_context(context):
        plt.scatter(g1,g2)
        plt.xlabel(group1)
        plt.ylabel(group2)
        if title is not None:
            plt.title(title)
        if fname is None:
            plt.savefig()
"""

def cutoffs_test(adata,by,stain_col,min_cutoff,max_cutoff,group1,group2,xlabel,ylabel, figure_dir,fname,num=50,values_to_include=None,class_map=None,context=None,title=None):
    cutoffs = np.linspace(min_cutoff,max_cutoff,num)
    g1=[]
    g2=[]
    df = adata.obs[adata.obs[by].isin(values_to_include)].copy()
    if class_map is not None:
            df['pseudo_group'] = df[by].map(class_map)
            by='pseudo_group'
    for cutoff in cutoffs:
        df['pass_cutoff'] = df[stain_col]>cutoff
        fracs_marked= df.groupby(by=by)['pass_cutoff'].sum()/(df[by].value_counts())
        g1.append(fracs_marked[group1])
        g2.append(fracs_marked[group2])
        # print(f'{fracs_marked[group1]:.2f}',f'{fracs_marked[group2]:.2f}',cutoff)
    if context is None:
        context = {'figure.figsize':(10,10),'font.sans-serif':"Arial",'font.family':"sans-serif","font.size":24,"figure.dpi":200}
    with plt.rc_context(context):
        plt.scatter(g1,g2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.gca().set_aspect('equal')
        if title is not None:
            plt.title(title)
        plt.savefig(os.path.join(figure_dir,fname),bbox_inches='tight')
    plt.show()

def cutoffs_bar(adata,by,stain_col,cutoff,xlabel,ylabel,figure_dir, fname, class_map=None,values_to_include=None, context=None,title=None):
    df = adata.obs.copy()
    if values_to_include is not None:
        df=df[df[by].isin(values_to_include)]
    df[by] = df[by].cat.remove_unused_categories()
    if class_map is not None:
        df['pseudo_group'] = df[by].map(class_map)
        by='pseudo_group'
    traced = df[stain_col] > cutoff
    df= pd.crosstab(traced, df[by])

    if context is None:
        context = {'font.sans-serif':"Arial",'font.family':"sans-serif","font.size":24,"figure.dpi":200}
    cmap = plt.get_cmap('Paired')

    with plt.rc_context(context):
        s = df.loc[True]/(df.loc[True]+df.loc[False])
        s=s.sort_values(ascending=False)
        ax = s.plot.bar(color=cmap(1))
        ax.set_xlabel('Type')
        ax.set_ylabel(f'Proportion of type marked')
        plt.title(title)
        plt.savefig(os.path.join(figure_dir,fname),bbox_inches='tight')
        plt.ylim(0,1)
        plt.show()

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
    c='blue',
    save_as=None,
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
        most_likely = np.argmax(confusion_matrix, axis=0)
        row_order = list(dict.fromkeys(most_likely)) # Note: If one type is most likely for multiple rows, this will get rid of the duplicates
        inv_dict = {v:k for k,v in test_dict.items()}
        unclear_assignment = set(test_dict.values()) - set(most_likely)
        row_order.extend(unclear_assignment)
        row_order = [inv_dict[i] for i in row_order]
        conf_df = conf_df.reindex(row_order)

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
        missing_cols = set(train_dict.keys()) - set(re_order_cols)
        re_order_cols.extend(list(missing_cols))
        conf_df=conf_df[re_order_cols]
    elif re_order:        
        conf_df=conf_df[re_order_cols]
    print(f"Reorder cols: {re_order_cols}")
    diagcm = conf_df.to_numpy()


        
    xticksactual = list(conf_df.columns)

    #dot_max = np.max(diagcm.flatten())
    dot_max = 1
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
    x_ticks = range(diagcm.shape[1])
    dot_ax.set_xticks(x_ticks)
    dot_ax.set_xticklabels(xticksactual, rotation=90)
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
    return diagcm, xticksactual, axs

def plot_mapping_new(test_labels, test_predlabels, test_dict, train_dict, xaxislabel, yaxislabel,re_order=None,re_order_cols = None,re_index = None, re_order_rows = None, save_as=None,c='blue'):
    
    ARI = adjusted_rand_score(labels_true = test_labels, 
                              labels_pred = test_predlabels)
    
           
    mappingconfmat, mappingxticks, mappingplot = plotConfusionMatrixNew(
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
    c=c,
    )    






        