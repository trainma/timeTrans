from numpy.lib.twodim_base import mask_indices
from sklearn.utils import shuffle
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
import torch.nn.functional as F
from torch.autograd import Variable
from models.positional_encoding import *

class AR(nn.Module):

    def __init__(self, window):

        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
    	# x: [batch, window, n_multiv]
        x = self.linear(x)  # x: [batch, n_multiv, 1]
        return x



class TransAm(nn.Module):
    def __init__(self, feature_size=256, num_layers=3, dropout=0.05,dec_seq_len=1,max_len=100,position='fixed',adding_module='default',batch_size=32,feature_dim=1,local=None):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.bs=batch_size
        self.feature_num=feature_dim
        # d_model 词嵌入维度
        self.d_model=feature_size
        self.dropout=dropout
        self.max_len=max_len
        self.input_project=nn.Linear(feature_dim,self.d_model)

        #---------whether use 1D conv-------------#
        self.local=context_embedding(self.d_model,self.d_model,1)

        #----------whether use learnable position encoding---------#
        self.pos_encoder = PositionalEncoding(feature_size)

        #----------whether transformer encoder layer with batch_norm---------#

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer=nn.TransformerDecoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        self.transformer_decoder=nn.TransformerDecoder(self.decoder_layer,num_layers=dec_seq_len)

        self.tmp_out=nn.Linear(feature_size,1)
        self.src_key_padding_mask = None
        self.transformer = nn.Transformer(d_model=self.d_model,
                                          nhead=8,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=dec_seq_len,
                                          dropout=dropout,)
        

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        """
        src:torch.Size([600, 32])->torch.Size([150, 32, 4]) [seq_len,batch_size,feature_dim]
        src_key_padding_mask:torch.Size([32, 150]) [batch_size,seq_len]
        src after linear projection:torch.Size([150, 32, 256]) [seq_len,batch_size,d_model]
        static_seq: torch.size([32,7])
        Encoder input:[seq_len,batch_size,d_model]
        """
        
        #src:[bz,timewindow,feature_dim] [16,168,321]
        src=src.permute(1,0,2)
        #tgt:[3,16,321]
        tgt = src[-3:, :, :]
         #取输入的最后一个时间切片作为decoder的一部分输入

        # tgt = src[-1:, :, :]
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 初始化掩码张量
        mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(device)


        src=self.input_project(src)* math.sqrt(self.d_model) # for input with feature dim=4
        tgt=self.input_project(tgt)* math.sqrt(self.d_model)
        #src:[168 16 64]
        #tgt:[3 16 64]

        # introducing casual conv on input time series => [150,32,256]
        # src:[168 16 64]->[16 64 168]
        # torch.Size([168, 16, 256])- torch.Size([16, 256, 168])->torch.Size([168, 16, 256])
        src=self.local(src.permute(1,2,0))
        src=src.permute(2,0,1)
        #torch.Size([168, 16, 256])->torch.Size([168, 16, pos_encoder.featuresize(256)])
        src = self.pos_encoder(src) # torch.Size([168,16,64 ])
        tgt = self.pos_encoder(tgt)#torch,Size(3 16 64)

        x = self.transformer(src=src,
                             tgt=tgt,
                             tgt_mask=mask)

        transformer_out=self.tmp_out(x)[0, :, :]
        return transformer_out

        # output = self.transformer_encoder(src, None, None)  # encoder output: torch.Size([150, 32, 256])
        
        # #------decoder+linear----#
        # d=src[-1,:].unsqueeze(0)
        # print('d:',d.shape)
        # tmp=self.transformer_decoder(d,output).squeeze(0)
        # final_out=self.tmp_out(tmp)
        # return final_out
if __name__ == '__main__':
    device =torch.device('cuda')
    model=TransAm(feature_dim=321).to(device)
    test_tensor=torch.randn(16,168,321,dtype=torch.float32).to(device)
    out=model(test_tensor)
    print(out.shape)
