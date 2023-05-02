import os
import torch
import argparse
import warnings
import time
from train import Train
from inits import load_data, fix_seed
from utils import UMAP, plot_weight_value
import pickle

warnings.filterwarnings("ignore")
os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3' 

parser = argparse.ArgumentParser(description='PyTorch implementation of spatial multi-omics data integration')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')  # 0.0001
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.') # 1500 Mouse Brain #1000 SPOTS_spleen_rep1 #700 Thymus
parser.add_argument('--weight_decay', type=float, default=0.0000, help='Weight for L2 loss on embedding matrix.')  # 5e-4
parser.add_argument('--dataset', type=str, default='Mouse_Embryo_E11', help='Dataset tested.')
parser.add_argument('--input', type=str, default='/home/yahui/anaconda3/work/SpatialGlue_omics/data/', help='Input path.')
parser.add_argument('--output', type=str, default='/home/yahui/anaconda3/work/SpatialGlue_omics/output/', help='output path.')
parser.add_argument('--random_seed', type=int, default=2022, help='Random seed') # 50
parser.add_argument('--dim_input', type=int, default=3000, help='Dimension of input features') # 100
parser.add_argument('--dim_output', type=int, default=64, help='Dimension of output features') # 64
parser.add_argument('--n_neighbors', type=int, default=6, help='Number of sampling neighbors') # 6
parser.add_argument('--n_clusters', type=int, default=9, help='Number of clustering') # mouse brain 15 thymus 9 spleen 5
args = parser.parse_args()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
t = time.time()
fix_seed(args.random_seed)

args.dataset = 'SPOTS_spleen_rep1'

if args.dataset == 'Thymus':
   args.n_clusters = 8
   args.epochs = 1500
elif args.dataset == 'Mouse_Brain':
   args.n_clusters = 15
   args.epochs = 1500
elif args.dataset == 'SPOTS_spleen_rep1':
   args.n_clusters = 6
   args.epochs = 900

print('>>>>>>>>>>>>>>>>>   {}   <<<<<<<<<<<<<<<<'.format(args.dataset))
     
data = load_data(args) 
adata_1, adata_2 = data['omics1'], data['omics2']
    
#start to train the model
trainer = Train(args, device, data) 
emb_1, emb_2, emb_combined, alpha_omics_1, alpha_omics_2, alpha_omics_1_2 = trainer.train()
print('time:', time.time()-t)

adata_1.obsm['emb'] = emb_1
adata_2.obsm['emb'] = emb_2
adata_1.obsm['emb_combined'] = emb_combined
adata_2.obsm['emb_combined'] = emb_combined

adata_1.obsm['alpha'] = alpha_omics_1_2

#adata_combined = adata_1.copy()
#adata_combined.obsm['RNA_feat'] = adata_1.obsm['feat']
#adata_combined.obsm['Protein_feat'] = adata_2.obsm['feat']
#adata_combined.obsm['RNA_latent'] = emb_1
#adata_combined.obsm['Protein_latent'] = emb_2
#adata_combined.obsm['emb_combined'] = emb_combined

#save_path = '/home/yahui/anaconda3/work/SpatialGlue_omics/output/' + args.dataset + '/'
#adata_combined.write_h5ad(save_path + 'adata_combined_new.h5ad')

#print('emb1:', emb_1)
#print('emb2:', emb_2)

adata_1.obsm['alpha'] = alpha_omics_1_2
 
result = {'omics1': adata_1, 'omics2': adata_2}
with open(args.output + args.dataset + '/result.pkl', 'wb') as file:
     pickle.dump(result, file) 
     
#print('alpha:', alpha)     

# umap
#adata_combined = UMAP(adata_1, adata_2, args)

alpha = []
#alpha.append(alpha_omics_1)
#alpha.append(alpha_omics_2)
alpha.append(alpha_omics_1_2)
# plotting weight value
plot_weight_value(alpha, adata_combined.obs['label_new_combined'].values, args)

#print('Time:', time.time() - t)

        
    
 



    
   
        
  
 
       
    
     


 
 
    
    
    

  
    
    
        