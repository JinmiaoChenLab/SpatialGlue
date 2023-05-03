import scipy
import anndata
import sklearn
import numpy as np
import scanpy as sc
import pandas as pd
import pickle
from typing import Optional, Union
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 

def read_data(args):
    """
    Reading adata
    """
    # read adata
    ## mouse spleen (SPOTS)
    adata_omics1 = sc.read_h5ad('/home/yahui/anaconda3/work/SpatialGlue_omics/data/SPOTS/Mouse_Spleen/Replicate1/adata_RNA.h5ad')
    adata_omics2 = sc.read_h5ad('/home/yahui/anaconda3/work/SpatialGlue_omics/data/SPOTS/Mouse_Spleen/Replicate1/adata_Pro.h5ad')  
    
    ## mouse brain (Spatial-ATAC-RNA-seq)
    #adata_omics1 = sc.read_h5ad('/home/yahui/anaconda3/work/SpatialGlue_omics/data/Rong_Fan/Mouse_Brain/Mouse_Brain_P22/Mouse_Brain_P22/adata_RNA.h5ad')
    #adata_omics2 = sc.read_h5ad('/home/yahui/anaconda3/work/SpatialGlue_omics/data/Rong_Fan/Mouse_Brain/Mouse_Brain_P22/Mouse_Brain_P22/adata_peaks_normalized.h5ad')
    
    # make feature names unique
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    # data pre-processing
    
    if args.datatype == 'SPOTS':  # mouse spleen 
      # RNA
      sc.pp.filter_genes(adata_omics1, min_cells=10)
      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(adata_omics1, target_sum=1e4)
      sc.pp.log1p(adata_omics1)
      sc.pp.pca(adata_omics1, use_highly_variable=True, n_comps=22) #22
      adata_omics1.obsm['feat'] = adata_omics1.obsm['X_pca'].copy()
    
      # Protein
      adata_omics2 = clr_normalize_each_cell(adata_omics2)
      sc.pp.pca(adata_omics2, n_comps=20)
      adata_omics2.obsm['feat'] = adata_omics2.obsm['X_pca'].copy()
      
    elif args.datatype == 'Stereo-CITE-seq':  
      # RNA
      sc.pp.filter_genes(adata_omics1, min_cells=10)
      sc.pp.filter_cells(adata_omics1, min_genes=80)

      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(adata_omics1, target_sum=1e4)
      sc.pp.log1p(adata_omics1)
      sc.pp.pca(adata_omics1, use_highly_variable=True, n_comps=50)
      adata_omics1.obsm['feat'] = adata_omics1.obsm['X_pca'].copy()
      
      # Protein
      sc.pp.filter_genes(adata_omics2, min_cells=50)
      proteins_to_remove = ['Mouse-IgD','Mouse-CD140a']
      proteins_to_keep = ~adata_omics2.var_names.isin(proteins_to_remove)
      adata_omics2 = adata_omics2[:, proteins_to_keep]
      
      adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
      
      adata_omics2 = clr_normalize_each_cell(adata_omics2)
      sc.pp.pca(adata_omics2, n_comps=50) #50
      adata_omics2.obsm['feat'] = adata_omics2.obsm['X_pca'].copy()
      
    elif args.datatype == 'Spatial-ATAC-RNA-seq':  
      # RNA
      sc.pp.filter_genes(adata_omics1, min_cells=10)
      sc.pp.filter_cells(adata_omics1, min_genes=200)
      
      sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(adata_omics1, target_sum=1e4)
      sc.pp.log1p(adata_omics1)
      sc.pp.pca(adata_omics1, use_highly_variable=True, n_comps=50)
      adata_omics1.obsm['feat'] = adata_omics1.obsm['X_pca'].copy()
      
      # ATAC
      adata_omics2 = adata_omics2[adata_omics1.obs_names].copy() # .obsm['X_lsi'] represents the dimension reduced feature
      adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()

    # construct cell neighbor graphs
    ################# spatial graph #################
    # omics1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = build_network(args, cell_position_omics1)
    adata_omics1.uns['adj_spatial'] = adj_omics1
    
    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = build_network(args, cell_position_omics2)
    adata_omics2.uns['adj_spatial'] = adj_omics2
    
    ################# feature graph #################
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2
    
    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}
    
    return data

def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     

def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode= "connectivity", metric="correlation", include_self=False):
    feature_graph_omics1=kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    feature_graph_omics2=kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)

    return feature_graph_omics1, feature_graph_omics2

def build_network(args, cell_position):
    nbrs = NearestNeighbors(n_neighbors=args.n_neighbors+1).fit(cell_position)  
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(args.n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj

def construct_graph(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf     
        
    
          
  
   