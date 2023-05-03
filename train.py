import torch
from model import Encoder_omics
from inits import preprocess_adj, preprocess_graph
from preprocess import construct_graph
import torch.nn.functional as F
from utils import plot_hist, plot_hist_multiple
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix

class Train:
    def __init__(self, args, device, data):
        self.args = args
        self.device = device
        self.data = data.copy()
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        
        # dimension of input feature
        self.args.dim_input1 = self.features_omics1.shape[1]
        self.args.dim_input2 = self.features_omics2.shape[1]
        self.args.dim_output1 = self.args.dim_output
        self.args.dim_output2 = self.args.dim_output
        
        ######################################## construct spatial graph ########################################
        self.adj_spatial_omics1 = self.adata_omics1.uns['adj_spatial']
        self.adj_spatial_omics1 = construct_graph(self.adj_spatial_omics1)
        self.adj_spatial_omics2 = self.adata_omics2.uns['adj_spatial']
        self.adj_spatial_omics2 = construct_graph(self.adj_spatial_omics2)
        
        self.adj_spatial_omics1 = self.adj_spatial_omics1.toarray()   # To ensure that adjacent matrix is symmetric
        self.adj_spatial_omics2 = self.adj_spatial_omics2.toarray()
        
        self.adj_spatial_omics1 = self.adj_spatial_omics1 + self.adj_spatial_omics1.T
        self.adj_spatial_omics1 = np.where(self.adj_spatial_omics1>1, 1, self.adj_spatial_omics1)
        self.adj_spatialomics2 = self.adj_spatial_omics2 + self.adj_spatial_omics2.T
        self.adj_spatial_omics2 = np.where(self.adj_spatial_omics2>1, 1, self.adj_spatial_omics2)
        
        # convert dense matrix to sparse matrix
        self.adj_spatial_omics1 = preprocess_graph(self.adj_spatial_omics1).to(self.device) # sparse adjacent matrix corresponding to spatial graph
        self.adj_spatial_omics2 = preprocess_graph(self.adj_spatial_omics2).to(self.device)
        
        ######################################## construct feature graph ########################################
        self.adj_feature_omics1 = torch.FloatTensor(self.adata_omics1.obsm['adj_feature'].copy().toarray())
        self.adj_feature_omics2 = torch.FloatTensor(self.adata_omics2.obsm['adj_feature'].copy().toarray())
        
        self.adj_feature_omics1 = self.adj_feature_omics1 + self.adj_feature_omics1.T
        self.adj_feature_omics1 = np.where(self.adj_feature_omics1>1, 1, self.adj_feature_omics1)
        self.adj_feature_omics2 = self.adj_feature_omics2 + self.adj_feature_omics2.T
        self.adj_feature_omics2 = np.where(self.adj_feature_omics2>1, 1, self.adj_feature_omics2)
        
        # convert dense matrix to sparse matrix
        self.adj_feature_omics1 = preprocess_graph(self.adj_feature_omics1).to(self.device) # sparse adjacent matrix corresponding to feature graph
        self.adj_feature_omics2 = preprocess_graph(self.adj_feature_omics2).to(self.device)
    
    def train(self):
        self.model = Encoder_omics(self.args.dim_input1, self.args.dim_output1, self.args.dim_input2, self.args.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.learning_rate, 
                                          weight_decay=self.args.weight_decay)
        #self.model.train()
        for epoch in tqdm(range(self.args.epochs)):
            self.model.train()
            #self.emb1_latent_within, self.emb2_latent_within, _, self.emb1_latent_recon, self.emb2_latent_recon, \
            #     self.emb_recon_omics1, self.emb_recon_omics2, self.alpha_omics_1, self.alpha_omics_2, self.alpha_omics_1_2, self.score1, self.score2 \
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)
            
            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
            
            # correspondence loss
            self.loss_corre_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
            self.loss_corre_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])
            
            # adversial loss
            self.label1 = torch.zeros(results['score_omics1'].size(0),).float().to(self.device)
            self.label2 = torch.ones(results['score_omics2'].size(0),).float().to(self.device)
            self.adversial_loss1 = F.mse_loss(results['score_omics1'], self.label1)
            self.adversial_loss2 = F.mse_loss(results['score_omics2'], self.label2)
            #self.loss_ad = 0.5*self.adversial_loss1 + 0.5*self.adversial_loss2
            self.loss_ad = self.adversial_loss1 + self.adversial_loss2
            
            if self.args.datatype == 'Spatial-ATAC-RNA-seq':
               loss = self.loss_recon_omics1 + 2.5*self.loss_recon_omics2 + self.loss_corre_omics1 + self.loss_corre_omics2 #+ self.loss_ad 
               print('self.loss_recon_omics1:', self.loss_recon_omics1)
               print('self.loss_recon_omics2:', 2.5*self.loss_recon_omics2)
               print('self.loss_corre_omics1:', self.loss_corre_omics1)
               print('self.loss_corre_omics2:', self.loss_corre_omics2)
               #print('self.loss_ad:', self.loss_ad)
               print('loss:', loss)
            elif self.args.datatype == 'SPOTS':  
               loss = self.loss_recon_omics1 + 50*self.loss_recon_omics2 + self.loss_corre_omics1 + 5*self.loss_corre_omics2 #+ self.loss_ad
               print('self.loss_recon_omics1:', self.loss_recon_omics1)
               print('self.loss_recon_omics2:', 50*self.loss_recon_omics2)
               print('self.loss_corre_omics1:', self.loss_corre_omics1)
               print('self.loss_corre_omics2:', 5*self.loss_corre_omics2)
               #print('self.loss_ad:', self.loss_ad)
               print('loss:', loss)
            elif self.args.datatype == 'Stereo-CITE-seq':
               loss = self.loss_recon_omics1 + 10*self.loss_recon_omics2 + self.loss_corre_omics1 + 10*self.loss_corre_omics2 #+ 10*self.loss_ad 
               print('self.loss_recon_omics1:', self.loss_recon_omics1)
               print('self.loss_recon_omics2:', 10*self.loss_recon_omics2)
               print('self.loss_corre_omics1:', self.loss_corre_omics1)
               print('self.loss_corre_omics2:', 10*self.loss_corre_omics2)
               #print('self.loss_ad:', 10*self.loss_ad)
               print('loss:', loss)   
              
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Model training finished!\n")    
    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)
 
        emb_omics1 = F.normalize(results['emb_recon_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_recon_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        return emb_omics1.detach().cpu().numpy(), emb_omics2.detach().cpu().numpy(), emb_combined.detach().cpu().numpy(), results['alpha'].detach().cpu().numpy()
    
    
    
        
    
    
      

    
        
    
    
