import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing

class SpatialGlue:
    def __init__(self, 
        data,
        datatype = 'SPOTS',
        device= torch.device('cpu'),
        random_seed = 2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=1500, 
        dim_input=3000,
        dim_output=64,
        ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.  
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.    
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
    
        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        
        #fix_seed(self.random_seed)
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
    
    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        #self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
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
            
            if self.datatype == 'Spatial-ATAC-RNA-seq':
               loss = self.loss_recon_omics1 + 2.5*self.loss_recon_omics2 + self.loss_corre_omics1 + self.loss_corre_omics2 #+ self.loss_ad 
               
            elif self.datatype == 'SPOTS':  
               loss = self.loss_recon_omics1 + 50*self.loss_recon_omics2 + self.loss_corre_omics1 + 5*self.loss_corre_omics2 #+ self.loss_ad
              
            elif self.datatype == 'Stereo-CITE-seq':
               loss = self.loss_recon_omics1 + 10*self.loss_recon_omics2 + self.loss_corre_omics1 + 10*self.loss_corre_omics2 #+ 10*self.loss_ad   
              
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Model training finished!\n")    
    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)
 
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'emb_latent_combined': emb_combined.detach().cpu().numpy(),
                  'alpha_omics1': results['alpha'].detach().cpu().numpy(),
                  'alpha_omics2': results['alpha'].detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy()}
        
        return output
    
    
    
        
    
    
      

    
        
    
    
