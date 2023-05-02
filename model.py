import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
#from torch_geometric.nn import GCNConv, GATConv
'''    
class Encoder_omics(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_feat1, out_feat1, in_feat2, out_feat2, dropout=0.0, act=F.relu):
        super(Encoder_omics, self).__init__()
        self.in_feat1 = in_feat1
        self.in_feat2 = in_feat2
        self.out_feat1 = out_feat1
        self.out_feat2 = out_feat2
        self.dropout = dropout
        self.act = act
        
        self.encoder_omics1 = Encoder(self.in_feat1, self.out_feat1)
        self.encoder_omics2 = Encoder(self.in_feat2, self.out_feat2)
        
        self.atten_omics1 = AttentionLayer(self.in_feat1, self.in_feat1)
        self.atten_omics2 = AttentionLayer(self.in_feat2, self.in_feat2)
        self.atten_cross = AttentionLayer(self.in_feat1, self.in_feat2)
        
        self.discriminator = Discriminator(self.in_feat1)
        
    def forward(self, feat1, feat2, adj1_1, adj1_2, adj2_1, adj2_2):
        # view1
        emb1_1_within = self.encoder_omics1(feat1, adj1_1)  # modality_veiw 
        emb2_1_within = self.encoder_omics2(feat2, adj2_1)
        
        # view2
        emb1_2_within = self.encoder_omics1(feat1, adj1_2)
        emb2_2_within = self.encoder_omics2(feat2, adj2_2)
        
        # within integration
        emb1_within, alpha_omics_1 = self.atten_omics1(emb1_1_within, emb1_2_within)
        emb2_within, alpha_omics_2 = self.atten_omics2(emb2_1_within, emb2_2_within)
        
        # cross integration
        emb_combined, alpha_omics_1_2 = self.atten_cross(emb1_within, emb2_within)
        
        emb1_cross = self.encoder_omics2(feat1, adj1_1) # encoding with each other
        emb2_cross = self.encoder_omics1(feat2, adj2_1)
        
        score1 = self.discriminator(emb1_within)
        score2 = self.discriminator(emb2_within)
        score1=torch.squeeze(score1, dim=1)
        score2=torch.squeeze(score2, dim=1)
        
        return emb1_within, emb2_within, emb1_cross, emb2_cross, emb_combined, alpha_omics_1, alpha_omics_2, alpha_omics_1_2, score1, score2    

class Encoder(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.weight2 = Parameter(torch.FloatTensor(self.out_feat, self.in_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight1)
        x = torch.spmm(adj, x)
        
        x = torch.mm(x, self.weight2)
        emb = torch.spmm(adj, x)
        
        return emb
'''
    
class Encoder_omics(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_feat1, out_feat1, in_feat2, out_feat2, dropout=0.0, act=F.relu):
        super(Encoder_omics, self).__init__()
        self.in_feat1 = in_feat1
        self.in_feat2 = in_feat2
        self.out_feat1 = out_feat1
        self.out_feat2 = out_feat2
        self.dropout = dropout
        self.act = act
        
        self.encoder_omics1 = Encoder(self.in_feat1, self.out_feat1)
        self.decoder_omics1 = Decoder(self.out_feat1, self.in_feat1)
        self.encoder_omics2 = Encoder(self.in_feat2, self.out_feat2)
        self.decoder_omics2 = Decoder(self.out_feat2, self.in_feat2)
        
        self.atten_omics1 = AttentionLayer(self.out_feat1, self.out_feat1)
        self.atten_omics2 = AttentionLayer(self.out_feat2, self.out_feat2)
        self.atten_cross = AttentionLayer(self.out_feat1, self.out_feat2)
        
        self.discriminator = Discriminator(self.out_feat1)
        
    def forward(self, feat1, feat2, adj1_1, adj1_2, adj2_1, adj2_2):
        # view1
        emb1_1_latent = self.encoder_omics1(feat1, adj1_1)  # modality_veiw
        emb1_1_recon  = self.decoder_omics1(emb1_1_latent, adj1_1)
        emb2_1_latent = self.encoder_omics2(feat2, adj2_1)
        emb2_1_recon  = self.decoder_omics2(emb2_1_latent, adj2_1)
        
        # view2
        emb1_2_latent = self.encoder_omics1(feat1, adj1_2)
        emb1_2_recon  = self.decoder_omics1(emb1_2_latent, adj1_2)
        emb2_2_latent = self.encoder_omics2(feat2, adj2_2)
        emb2_2_recon  = self.decoder_omics2(emb2_2_latent, adj2_2)
        
        # within integration
        emb1_latent_within, alpha_omics_1 = self.atten_omics1(emb1_1_latent, emb1_2_latent)
        emb2_latent_within, alpha_omics_2 = self.atten_omics2(emb2_1_latent, emb2_2_latent)
        
        # cross integration
        emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb1_latent_within, emb2_latent_within)
        
        # reconstruct feature via two modality decoders, respectively
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj1_1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj2_1)
        
        emb1_latent_recon = self.encoder_omics2(self.decoder_omics2(emb1_latent_within, adj2_1), adj2_1) # consistent encoding  # dim=64
        emb2_latent_recon = self.encoder_omics1(self.decoder_omics1(emb2_latent_within, adj1_1), adj1_1)
        
        score1 = self.discriminator(emb1_latent_within)
        score2 = self.discriminator(emb2_latent_within)
        score1=torch.squeeze(score1, dim=1)
        score2=torch.squeeze(score2, dim=1)
        
        return emb1_latent_within, emb2_latent_within, emb_latent_combined, emb1_latent_recon, emb2_latent_recon, emb_recon_omics1, emb_recon_omics2, alpha_omics_1, alpha_omics_2, alpha_omics_1_2, score1, score2 
    
class Omics1(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Omics1, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.encoder = Encoder(self.in_feat, self.out_feat)
        self.decoder = Decoder(self.out_feat, self.in_feat)
        
    def forward(self, feat, adj):
        x_latent = self.encoder(feat, adj)
        
        x_recon = self.decoder(adj, x_latent)
        
        return x_latent, x_recon  

class Omics2(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Omics2, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.encoder = Encoder(self.in_feat, self.out_feat)
        self.decoder = Decoder(self.out_feat, self.in_feat)
        
    def forward(self, feat, adj):
        x_latent = self.encoder(feat, adj)
        
        x_recon = self.decoder(adj, x_latent)
        
        return x_latent, x_recon    

class Encoder(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        
        return x
    
class Decoder(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        
        return x      
    
class Discriminator(nn.Module):
    """Latent space discriminator"""
    def __init__(self, dim_input, n_hidden=50, n_out=1):
        super(Discriminator, self).__init__()
        self.dim_input = dim_input
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(dim_input, n_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_hidden, 2*n_hidden),
            nn.LeakyReLU(inplace=True),
            #nn.Linear(n_hidden, n_hidden),
            #nn.ReLU(inplace=True),
            #nn.Linear(n_hidden, n_hidden),
            #nn.ReLU(inplace=True),
            #nn.Linear(n_hidden, n_hidden),
            #nn.ReLU(inplace=True),
            nn.Linear(2*n_hidden,n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)             

class AttentionLayer(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  #[5,2]
        #print('alpha:', self.alpha)
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha      
    
  
    

        
        
           
    
