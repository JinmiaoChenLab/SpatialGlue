import os
import pickle
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'    
    
def UMAP(adata_s, adata_t, args, size1=10, size2=20, resolution=0.2): #10, 20
    
    if args.dataset in ['SPOTS_spleen_rep1', 'SPOTS_spleen_rep2']:
      # rotate image
      adata_s.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata_s.obsm['spatial'])).T).T).T
      adata_t.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata_t.obsm['spatial'])).T).T).T
      adata_s.obsm['spatial'][:,1] = -1*adata_s.obsm['spatial'][:,1]     
      adata_t.obsm['spatial'][:,1] = -1*adata_t.obsm['spatial'][:,1]
    elif args.dataset in ['Thymus']:  
      adata_s.obsm['spatial'][:,1] = -1*adata_s.obsm['spatial'][:,1]     
      adata_t.obsm['spatial'][:,1] = -1*adata_t.obsm['spatial'][:,1]

    if args.dataset == 'Mouse_Brain':
       adata_s.obsm['emb_pca'] = sc.tl.pca(adata_s.obsm['emb'], n_comps=20)
       #adata_t.obsm['emb_pca'] = adata_t.obsm['emb']
       adata_t.obsm['emb_pca'] = sc.tl.pca(adata_t.obsm['emb'], n_comps=10)
       adata_s.obsm['origi_pca'] = sc.tl.pca(adata_s.obsm['feat'], n_comps=20)
       adata_t.obsm['origi_pca'] = adata_t.obsm['feat']
    elif args.dataset == 'SPOTS_spleen_rep1': 
       adata_s.obsm['emb_pca'] = sc.tl.pca(adata_s.obsm['emb'], n_comps=20)
       #adata_t.obsm['emb_pca'] = adata_t.obsm['emb']
       adata_t.obsm['emb_pca'] = sc.tl.pca(adata_t.obsm['emb'], n_comps=10)
       adata_s.obsm['origi_pca'] = adata_s.obsm['feat']
       adata_t.obsm['origi_pca'] = adata_t.obsm['feat']
    elif args.dataset == 'Thymus': 
       adata_s.obsm['emb_pca'] = sc.tl.pca(adata_s.obsm['emb'], n_comps=10) #20
       #adata_t.obsm['emb_pca'] = adata_t.obsm['emb']
       adata_t.obsm['emb_pca'] = sc.tl.pca(adata_t.obsm['emb'], n_comps=10)
       adata_s.obsm['origi_pca'] = adata_s.obsm['feat']
       adata_t.obsm['origi_pca'] = adata_t.obsm['feat']   
    
    clustering(adata_s, keys='emb_pca', add_keys='label_new', n_clusters=args.n_clusters)
    clustering(adata_t, keys='emb_pca', add_keys='label_new', n_clusters=args.n_clusters)
    
    clustering(adata_s, keys='origi_pca', add_keys='label_origi', n_clusters=args.n_clusters)
    clustering(adata_t, keys='origi_pca', add_keys='label_origi', n_clusters=args.n_clusters)
    
    # -------------------   plotting UMAP   ----------------------------
    fig, ax_list = plt.subplots(2, 5, figsize=(11, 5))
    #fig, ax_list = plt.subplots(2, 5, figsize=(8, 5)) # breast cancer
    ## UMAP on original feature
    if args.dataset == 'Mouse_Brain':
       sc.pp.neighbors(adata_s, use_rep='feat', n_neighbors=10)
    else:   
       sc.pp.neighbors(adata_s, use_rep='origi_pca', n_neighbors=10)
    sc.tl.umap(adata_s)
    #sc.tl.leiden(adata_s, resolution=resolution, key_added='leiden_origi')
    sc.pl.umap(adata_s, color='label_origi', ax=ax_list[0, 0], title='mRNA_origi', s=size1, show=False)
    adata_ss = adata_s.copy()
    
    if args.dataset == 'Mouse_Brain':
       sc.pp.neighbors(adata_t, use_rep='feat', n_neighbors=10)
    else:
       sc.pp.neighbors(adata_t, use_rep='origi_pca', n_neighbors=10) 
    sc.tl.umap(adata_t)
    #sc.tl.leiden(adata_t, resolution=resolution, key_added='label_new')
    sc.pl.umap(adata_t, color='label_origi', ax=ax_list[1, 0], title='Protein_origi', s=size1, show=False)
    adata_tt = adata_t.copy()
    
    ## UMAP on latent representation
    sc.pp.neighbors(adata_s, use_rep='emb_pca', n_neighbors=10)
    sc.tl.umap(adata_s)
    #sc.tl.leiden(adata_s, resolution=resolution, key_added='leiden_new')
    sc.pl.umap(adata_s, color='label_new', ax=ax_list[0, 1], title='mRNA_new', s=size1, show=False)
    
    sc.pp.neighbors(adata_t, use_rep='emb_pca', n_neighbors=10)
    sc.tl.umap(adata_t)
    #sc.tl.leiden(adata_t, resolution=resolution, key_added='leiden_new')
    sc.pl.umap(adata_t, color='label_new', ax=ax_list[1, 1], title='Protein_new', s=size1, show=False)
    
    # -------------------   plotting UMAP based on original and latent features   ----------------------------   
    # conbined feature
    adata_combined = adata_s.copy()
    
    print('adata_combined:', adata_combined)
    
    # add original RNA and Protein features
    adata_combined.obsm['RNA_feat'] = adata_s.obsm['feat']
    adata_combined.obsm['Pro_feat'] = adata_t.obsm['feat']
    
    adata_combined.obsm['RNA_latent'] = adata_s.obsm['emb']
    adata_combined.obsm['Pro_latent'] = adata_t.obsm['emb']
    
   
    adata_combined.obsm['emb_combined_pca'] = sc.tl.pca(adata_combined.obsm['emb_combined'], n_comps=20)
    clustering(adata_combined, keys='emb_combined_pca', add_keys='label_new_combined', n_clusters=args.n_clusters)
    sc.pp.neighbors(adata_combined, use_rep='emb_combined_pca', n_neighbors=10)
    sc.tl.umap(adata_combined)
    #sc.tl.leiden(adata_combined, resolution=resolution, key_added='leiden_combined')
    sc.pl.umap(adata_combined, color='label_new_combined', ax=ax_list[0, 4], title='integrated', s=size2, show=False)
    sc.pl.embedding(adata_combined, basis='spatial', color='label_new_combined', ax=ax_list[1, 4], title='integrated', s=size2, show=False)
    

    adata_s.obs['label_new_combined'] = adata_combined.obs['label_new_combined']
    
    # -------------------   plotting UMAP based on original and latent features   ----------------------------
    #adata_s.obsm['spatial'][:, 0] = -1*adata_s.obsm['spatial'][:, 0]
    #adata_s.obsm['spatial'][:, 1] = -1*adata_s.obsm['spatial'][:, 1]
    
    #adata_t.obsm['spatial'][:, 0] = -1*adata_t.obsm['spatial'][:, 0]
    #adata_t.obsm['spatial'][:, 1] = -1*adata_t.obsm['spatial'][:, 1]
    
    sc.pl.embedding(adata_s, basis='spatial', color='label_origi', ax=ax_list[0, 2], title='mRNA_origi', s=size2, show=False)
    sc.pl.embedding(adata_t, basis='spatial', color='label_origi', ax=ax_list[1, 2], title='Protein_origi', s=size2, show=False)
    
    sc.pl.embedding(adata_s, basis='spatial', color='label_new', ax=ax_list[0, 3], title='mRNA_new', s=size2, show=False)
    sc.pl.embedding(adata_t, basis='spatial', color='label_new', ax=ax_list[1, 3], title='Protein_new', s=size2, show=False)
    
    #adata_ss.obs['label_origi'] = adata_combined.obs['label_new_combined']
    #adata_tt.obs['label_origi'] = adata_combined.obs['label_new_combined']
    #sc.pl.umap(adata_ss, color='label_origi', ax=ax_list[0, 0], title='mRNA_origi', s=size1, show=False)
    #sc.pl.umap(adata_tt, color='label_origi', ax=ax_list[1, 0], title='Protein_origi', s=size1, show=False)
    
    
    #adata_s.obs['label_new'] = adata_combined.obs['label_new_combined']
    #adata_t.obs['label_new'] = adata_combined.obs['label_new_combined']
    #sc.pl.umap(adata_s, color='label_new', ax=ax_list[0, 1], title='mRNA_new', s=size1, show=False)
    #sc.pl.umap(adata_t, color='label_new', ax=ax_list[1, 1], title='Protein_new', s=size1, show=False)
    
    # save adata_combined
    save_path = '/home/yahui/anaconda3/work/SpatialGlue_omics/output/' + args.dataset + '/'
    adata_combined.write_h5ad(save_path + 'adata_combined.h5ad')
    
    ax_list[0, 0].get_legend().remove()
    ax_list[1, 0].get_legend().remove()
    ax_list[0, 1].get_legend().remove()
    ax_list[1, 1].get_legend().remove()
    ax_list[0, 2].get_legend().remove()
    ax_list[1, 2].get_legend().remove()
    ax_list[0, 3].get_legend().remove()
    ax_list[1, 3].get_legend().remove()
    #ax_list[0, 4].get_legend().remove()
    #ax_list[1, 4].get_legend().remove()
    #ax_list[0, 5].get_legend().remove()
    #ax_list[1, 5].get_legend().remove()
    
    plt.tight_layout(w_pad=0.3)
    plt.show()
    
    #save_path = '/home/yahui/anaconda3/work/SpatialGlue_omics/output/' + args.dataset + '/'
    #plt.savefig(save_path + args.dataset + '.jpg', bbox_inches='tight', dpi=300)
    
    return adata_combined

def plot_hist_multiple(hist):
    fig=plt.figure(figsize=(8,4), dpi=100)
    i = 1
    for key in hist.keys():
       lost = hist[key]
       values = np.array(lost)
       size = values.size
       values = values.flatten()
       ax = fig.add_subplot(2,4,i)
       ax.plot(np.arange(0, size), values)
       ax.title.set_text(str(key))
       i += 1
    #plt.layout(w_pad=0.01)
    #plt.layout(h_pad=0.01)
    plt.tight_layout()
    plt.show()   
        
    
def plot_hist(lost):
    #ax = plt
    #print('Plotting loss')
    #print(lost)
    values = np.array(lost)
    size = values.size
    values = values.flatten()
    #values = np.log10(values.flatten())
    plt.plot(np.arange(0, size), values)
    plt.show()

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, keys='emb_pca', add_keys='label', n_clusters=10):
    #pca = PCA(n_components=20, random_state=42) 
    #embedding = pca.fit_transform(adata.obsm['emb_sp'])
    #adata.obsm['emb_sp_pca'] = embedding
    adata = mclust_R(adata, used_obsm=keys, num_cluster=n_clusters)
    adata.obs[add_keys] = adata.obs['mclust']
    #adata = refine_label(adata, args)
    
    #return adata
  
def calculate_distance(pos):
  num_cell = pos.shape[0]
  distance_matrix = np.zeros([num_cell, num_cell])
  for i in range(num_cell):
    x = pos[i]
    for j in range(i+1, num_cell):
        y = pos[j]
        d = np.sqrt(np.sum(np.square(x-y)))
        distance_matrix[i, j] = d
        distance_matrix[j, i] = d
  return distance_matrix

def refine_label(adata, args):
    print('n_neigh:', args.n_circle)
    n_neigh = args.n_circle #50
    new_type = []
    old_type = adata.obs['label'].values
    
    #read distance
    input_path = args.input + args.dataset + '/distance/'
    
    if not os.path.exists(input_path):
       os.makedirs(input_path)
    
    if os.path.isfile(input_path + 'distance' + '.pkl'):
       print('Distance file exists.') 
       distance = pickle.load(open(input_path + 'distance' + '.pkl', 'rb'))
    else: 
       cell_position = adata.obsm['spatial'] 
       distance = calculate_distance(cell_position)
       with open(input_path + 'distance' + '.pkl', 'wb') as file:
           pickle.dump(distance, file)
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    adata.obs['label_refined'] = np.array(new_type)
    
    return adata

def plot_weight_value(alpha, label, args):
  import pandas as pd  
  #fig, ax_list = plt.subplots(1, 3, figsize=(11, 3))
  for i, val in enumerate(alpha):
    df = pd.DataFrame(columns=['mRNA','protein','label'])  
    df['mRNA'], df['protein'] = val[:, 0], val[:, 1]
    df['label'] = label
    df = df.set_index('label').stack().reset_index()
    df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
    ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
    ax.set_title('mRNA vs protein')
  
  plt.tight_layout(w_pad=0.05)
  plt.show()
  #path = '/home/yahui/anaconda3/work/SpatialGlue_omics/output/' + args.dataset + '/alpha/' 
  #plt.savefig(path + 'alpha.jpg', bbox_inches='tight', dpi=300)    
'''
### UMAP_omics
## read data
import argparse
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--dataset', type=str, default='SPOTS', help='Dataset tested.')
parser.add_argument('--n_clusters', type=int, default=5, help='Number of clustering') 
args = parser.parse_args()

args.dataset = 'SPOTS_spleen_rep1'
path = '/home/yahui/anaconda3/work/SpatialGlue_omics/output/' + args.dataset + '/'
result = pickle.load(open(path + 'result.pkl', 'rb'))
adata_1, adata_2 = result['omics1'], result['omics2']
print('adata_1:', adata_1)
print('adata_2:', adata_2)

UMAP(adata_1, adata_2, args=args)
'''