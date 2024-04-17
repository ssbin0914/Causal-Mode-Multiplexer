import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.att_layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attention1 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.attention2 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.attention3 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.attention4 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)

#        self.out_att = GraphAttentionLayer(nhid, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.out_att = GraphAttentionLayer(nhid*4, nclass, dropout=dropout, alpha=alpha, concat=False)
    def forward(self, batch_size, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat((self.attention1(batch_size,x,adj), self.attention2(batch_size,x,adj), self.attention3(batch_size,x,adj), self.attention4(batch_size,x,adj)), 2)
        x = F.dropout(x, self.dropout, training=self.training)
#        x = self.attention1(batch_size,x,adj)
        x = F.elu(self.out_att(batch_size,x,adj))
        x = x.permute(0,2,1).contiguous()
        return x


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
