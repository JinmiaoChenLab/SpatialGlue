import torch
from model import Encoder_omics
from inits import get_edge_index, preprocess_adj, preprocess_graph
#from torch import nn
from preprocess import construct_graph
import torch.nn.functional as F
from utils import plot_hist, plot_hist_multiple
#import pickle
from tqdm import tqdm
#import scipy.sparse as sp
import numpy as np
#import ot
from scipy.sparse import coo_matrix

class Train:
    def __init__(self, args, device, data):
        self.args = args
        self.device = device
        self.data = data.copy()
        self.adata_1 = self.data['omics1']
        self.adata_2 = self.data['omics2']
        
        self.n_omics1 = self.adata_1.n_obs
        self.n_omics2 = self.adata_2.n_obs
        
        # feature
        self.features_1 = torch.FloatTensor(self.adata_1.obsm['feat'].copy()).to(self.device)
        self.features_2 = torch.FloatTensor(self.adata_2.obsm['feat'].copy()).to(self.device)
        
        # dimension of input feature
        self.args.dim_input1 = self.features_1.shape[1]
        self.args.dim_input2 = self.features_2.shape[1]
        self.args.dim_output1 = self.args.dim_output
        self.args.dim_output2 = self.args.dim_output
        
        ######################################## adj-view1 ########################################
        self.adj_1 = self.adata_1.uns['adj']
        self.adj_1 = construct_graph(self.adj_1)
        self.adj_2 = self.adata_2.uns['adj']
        self.adj_2 = construct_graph(self.adj_2)
        
        self.graph_1 = self.adj_1.toarray()   # ensure adjacent matrix is symmetric
        self.graph_2 = self.adj_2.toarray()
        self.graph_1 = self.graph_1 + self.graph_1.T
        self.adj_1 = np.where(self.graph_1>1, 1, self.graph_1)
        self.graph_2 = self.graph_2 + self.graph_2.T
        self.adj_2 = np.where(self.graph_2>1, 1, self.graph_2)
        
        self.adj1_1 = preprocess_graph(self.adj_1).to(self.device) # view 1 (by position)
        self.adj2_1 = preprocess_graph(self.adj_2).to(self.device)
        
        # pytorch version
        #self.adj_1 = get_edge_index(self.graph_1).to(self.device)
        #self.adj_2 = get_edge_index(self.graph_2).to(self.device)
        
        ######################################## adj-view2 ########################################
        self.graph_1 = torch.FloatTensor(self.adata_1.obsm['graph_feat'].copy().toarray())
        self.graph_2 = torch.FloatTensor(self.adata_2.obsm['graph_feat'].copy().toarray())
        
        self.graph_1 = self.graph_1 + self.graph_1.T
        self.adj_1 = np.where(self.graph_1>1, 1, self.graph_1)
        self.graph_2 = self.graph_2 + self.graph_2.T
        self.adj_2 = np.where(self.graph_2>1, 1, self.graph_2)
        
        # Sparse version
        self.adj1_2 = preprocess_graph(self.adj_1).to(self.device) # view 2 (by feature)
        self.adj2_2 = preprocess_graph(self.adj_2).to(self.device)
        
    '''    
    def train(self):
        self.model = Encoder_omics(self.args.dim_input1, self.args.dim_output1, self.args.dim_input2, self.args.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.learning_rate, 
                                          weight_decay=self.args.weight_decay)
        self.model.train()
        hist_overall, hist_loss_1, hist_loss_2, hist_loss_11_, \
           hist_loss_22_, hist_loss_11, hist_loss_22, hist_loss_ad = [], [], [], [], [], [], [], []
        hist = dict()
        for epoch in tqdm(range(self.args.epochs)):
            self.model.train()
            self.emb1_within, self.emb2_within, self.emb1_cross, self.emb2_cross, \
            self.emb_combined, _, _, _, self.score1, self.score2 \
                 = self.model(self.features_1, self.features_2, self.adj1_1, self.adj1_2, self.adj2_1, self.adj2_2)
            
            # recon loss
            self.loss_1 = F.mse_loss(self.features_1, self.emb1_within)
            self.loss_2 = F.mse_loss(self.features_2, self.emb2_within)
            
            self.loss_11_ = F.mse_loss(self.features_1, self.emb1_cross)
            self.loss_22_ = F.mse_loss(self.features_2, self.emb2_cross)
            
            self.loss_11 = F.mse_loss(self.features_1, self.emb_combined)
            self.loss_22 = F.mse_loss(self.features_2, self.emb_combined)
            
            # adversial loss
            self.label1 = torch.zeros(self.score1.size(0),).float().to(self.device)
            self.label2 = torch.ones(self.score2.size(0),).float().to(self.device)
            self.adversial_loss1 = F.mse_loss(self.score1, self.label1)
            self.adversial_loss2 = F.mse_loss(self.score2, self.label2)
            self.loss_ad = 0.5*self.adversial_loss1 + 0.5*self.adversial_loss2
            
            
            #loss = self.loss_11 + 10*self.loss_22 + 10*self.loss_ad + self.loss_1 + 10*self.loss_2   #Thymus 
            #loss = self.loss_11 + 20*self.loss_22 + 10*self.loss_ad + self.loss_1 + 10*self.loss_2   # SPOTS
            loss = self.loss_11 + 20*self.loss_22 + 10*self.loss_ad + self.loss_1 + 10*self.loss_2 #+ self.loss_11_ + 20*self.loss_22_
            #loss = self.loss_11 + 10*self.loss_22  #rep 1
            #loss = self.loss_11 + self.loss_22 #+  self.loss_1 + self.loss_2 + self.loss_11_ + self.loss_22_ + self.loss_ad
            print('self.loss_11:', self.loss_11)
            print('self.loss_22:', 20*self.loss_22)
            print('self.loss_ad:', 10*self.loss_ad)
            print('self.loss_11_:', self.loss_11_)
            print('self.loss_22_:', 10*self.loss_22_)
            print('loss:', loss)

            hist_overall.append(loss.data.cpu().numpy())
            hist_loss_1.append(self.loss_1.data.cpu().numpy())
            hist_loss_2.append(self.loss_2.data.cpu().numpy())
            hist_loss_11_.append(self.loss_11_.data.cpu().numpy())
            hist_loss_22_.append(self.loss_22_.data.cpu().numpy())
            hist_loss_11.append(self.loss_11.data.cpu().numpy())
            hist_loss_22.append(self.loss_22.data.cpu().numpy())
            hist_loss_ad.append(self.loss_ad.data.cpu().numpy())
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            
        # plot training loss
        hist = {'loss_1': hist_loss_1, 'loss_2': hist_loss_2, 'loss_11_': hist_loss_11_,
                'loss_22_': hist_loss_22_, 'loss_11': hist_loss_11, 'loss_22': hist_loss_22, 'loss_ad': hist_loss_ad, 'loss': hist_overall}
        
        #plot_hist(hist)
        plot_hist_multiple(hist)
        print("Training finished!\n")    
        
        with torch.no_grad():
          self.model.eval()
          emb_1, emb_2, _, _, emb_combined, alpha_omics_1, alpha_omics_2, alpha_omics_1_2, _, _ = self.model(self.features_1, self.features_2, self.adj1_1, self.adj1_2, self.adj2_1, self.adj2_2)
          
        emb_1 = F.normalize(emb_1, p=2, eps=1e-12, dim=1)  
        emb_2 = F.normalize(emb_2, p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(emb_combined, p=2, eps=1e-12, dim=1)
        
        return emb_1.detach().cpu().numpy(), emb_2.detach().cpu().numpy(), emb_combined.detach().cpu().numpy(), alpha_omics_1.detach().cpu().numpy(), alpha_omics_2.detach().cpu().numpy(), alpha_omics_1_2.detach().cpu().numpy()
    '''
    
    def train(self):
        self.model = Encoder_omics(self.args.dim_input1, self.args.dim_output1, self.args.dim_input2, self.args.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.learning_rate, 
                                          weight_decay=self.args.weight_decay)
        self.model.train()
        hist_overall, hist_1, hist_2, hist_3, \
           hist_4, hist_5, hist_6, hist_7 = [], [], [], [], [], [], [], []
        hist = dict()
        for epoch in tqdm(range(self.args.epochs)):
            self.model.train()
            self.emb1_latent_within, self.emb2_latent_within, _, self.emb1_latent_recon, self.emb2_latent_recon, \
                 self.emb_recon_omics1, self.emb_recon_omics2, self.alpha_omics_1, self.alpha_omics_2, self.alpha_omics_1_2, self.score1, self.score2 \
                 = self.model(self.features_1, self.features_2, self.adj1_1, self.adj1_2, self.adj2_1, self.adj2_2)
            
            # reconstruction loss
            self.loss_recon_1 = F.mse_loss(self.features_1, self.emb_recon_omics1)
            self.loss_recon_2 = F.mse_loss(self.features_2, self.emb_recon_omics2)
            
            # consistency loss
            self.loss_latent_recon_1 = F.mse_loss(self.emb1_latent_within, self.emb1_latent_recon)
            self.loss_latent_recon_2 = F.mse_loss(self.emb2_latent_within, self.emb2_latent_recon)
            
            # adversial loss
            self.label1 = torch.zeros(self.score1.size(0),).float().to(self.device)
            self.label2 = torch.ones(self.score2.size(0),).float().to(self.device)
            self.adversial_loss1 = F.mse_loss(self.score1, self.label1)
            self.adversial_loss2 = F.mse_loss(self.score2, self.label2)
            #self.loss_ad = 0.5*self.adversial_loss1 + 0.5*self.adversial_loss2
            self.loss_ad = self.adversial_loss1 + self.adversial_loss2
            
            if self.args.dataset == 'Mouse_Brain':
               loss = self.loss_recon_1 + 2.5*self.loss_recon_2 + self.loss_latent_recon_1 + self.loss_latent_recon_2 #+ self.loss_ad # Rong Fan Mouse Brain
               print('self.loss_recon_1:', self.loss_recon_1)
               print('self.loss_recon_2:', 2.5*self.loss_recon_2)
               print('self.loss_latent_recon_1:', self.loss_latent_recon_1)
               print('self.loss_latent_recon_2:', self.loss_latent_recon_2)
               #print('self.loss_ad:', self.loss_ad)
               print('loss:', loss)
            elif self.args.dataset == 'SPOTS_spleen_rep1':
               #loss = self.loss_recon_1 + 50*self.loss_recon_2 + self.loss_latent_recon_1 + 5*self.loss_latent_recon_2  
               loss = self.loss_recon_1 + 50*self.loss_recon_2 + self.loss_latent_recon_1 + 5*self.loss_latent_recon_2
               print('self.loss_recon_1:', self.loss_recon_1)
               print('self.loss_recon_2:', 50*self.loss_recon_2)
               print('self.loss_latent_recon_1:', self.loss_latent_recon_1)
               print('self.loss_latent_recon_2:', 5*self.loss_latent_recon_2)
               #print('self.loss_ad:', self.loss_ad)
               print('loss:', loss)
            elif self.args.dataset == 'Thymus':
               #loss = self.loss_recon_1 + self.loss_recon_2 + self.loss_latent_recon_1 + 10*self.loss_latent_recon_2 + self.loss_ad
               loss = self.loss_recon_1 + 10*self.loss_recon_2 + self.loss_latent_recon_1 + 10*self.loss_latent_recon_2 #+ 10*self.loss_ad 
               print('self.loss_recon_1:', self.loss_recon_1)
               print('self.loss_recon_2:', 10*self.loss_recon_2)
               print('self.loss_latent_recon_1:', self.loss_latent_recon_1)
               print('self.loss_latent_recon_2:', 10*self.loss_latent_recon_2)
               #print('self.loss_ad:', 10*self.loss_ad)
               print('loss:', loss)   
               

            hist_overall.append(loss.data.cpu().numpy())
            hist_1.append(self.loss_recon_1.data.cpu().numpy())
            hist_2.append(self.loss_recon_2.data.cpu().numpy())
            hist_3.append(self.loss_latent_recon_1.data.cpu().numpy())
            hist_4.append(self.loss_latent_recon_2.data.cpu().numpy())
            hist_5.append(self.loss_ad.data.cpu().numpy())
            #hist_loss_22.append(self.loss_22.data.cpu().numpy())
            #hist_loss_ad.append(self.loss_ad.data.cpu().numpy())
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            
        # plot training loss
        #hist = {'loss_1': hist_loss_1, 'loss_2': hist_loss_2, 'loss_11_': hist_loss_11_,
        #        'loss_22_': hist_loss_22_, 'loss_11': hist_loss_11, 'loss_22': hist_loss_22, 'loss_ad': hist_loss_ad, 'loss': hist_overall}
        
        hist = {'loss_recon_1': hist_1, 'loss_recon_2': hist_2, 'loss_latent_recon_1': hist_3, 'loss_latent_recon_2': hist_4, \
                'loss_ad': hist_5, 'loss': hist_overall}
        
        #plot_hist(hist)
        plot_hist_multiple(hist)
        print("Model training finished!\n")    
        
        '''
        with torch.no_grad():
          self.model.eval()
          emb_1, emb_2, _, _, emb_combined, alpha_omics_1, alpha_omics_2, alpha_omics_1_2, _, _ = self.model(self.features_1, self.features_2, self.adj1_1, self.adj1_2, self.adj2_1, self.adj2_2)
          
        emb_1 = F.normalize(emb_1, p=2, eps=1e-12, dim=1)  
        emb_2 = F.normalize(emb_2, p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(emb_combined, p=2, eps=1e-12, dim=1)
        
        return emb_1.detach().cpu().numpy(), emb_2.detach().cpu().numpy(), emb_combined.detach().cpu().numpy(), alpha_omics_1.detach().cpu().numpy(), alpha_omics_2.detach().cpu().numpy(), alpha_omics_1_2.detach().cpu().numpy()
        '''
    
        with torch.no_grad():
          self.model.eval()
          emb_1, emb_2, emb_combined, _, _, _, _, alpha_omics_1, alpha_omics_2, alpha_omics_1_2, _, _ = self.model(self.features_1, self.features_2, self.adj1_1, self.adj1_2, self.adj2_1, self.adj2_2)
 
        emb_1 = F.normalize(emb_1, p=2, eps=1e-12, dim=1)  
        emb_2 = F.normalize(emb_2, p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(emb_combined, p=2, eps=1e-12, dim=1)
        
        return emb_1.detach().cpu().numpy(), emb_2.detach().cpu().numpy(), emb_combined.detach().cpu().numpy(), alpha_omics_1.detach().cpu().numpy(), alpha_omics_2.detach().cpu().numpy(), alpha_omics_1_2.detach().cpu().numpy()
    
    
    
        
    
    
      

    
        
    
    
