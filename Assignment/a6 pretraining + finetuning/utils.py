import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from transformers import AutoModel, AutoTokenizer
import random
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from typing import List, Dict, Tuple, Type, Union
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import RobertaTokenizer
import random


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=None):
        super().__init__()

        if temp !=None:
            self.temp = temp
        else: 

            self.temp = 1.0
        # among feature last dim 
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        
        return  self.cos(x, y) / self.temp


def create_supervised_pair(h,labels,debug:bool=False):
    

    """
     h - (batch_size, seq_len, hidden_dim)
     label - (batch_size) 
    
    """
    # params
    h_i = [] # pos pairs
    h_j = [] # neg pairs
    skips = []
    idx_i = [] # index yi = yj

    # proof done 
    # masking correct
    # pair of concat correct

    T = 0 # the numbers of pairs sample
    
    for idx, label in enumerate(labels): 

        if idx in skips:
            continue

        mask = label == np.array(labels)
        
        # dont include current sample 
        mask[idx] = False 
        
        # check is they have pos pair 
        if np.count_nonzero(mask) >= 1:
            

            idxs_arr = np.arange(len(labels))
            # each h_i and h_j :  (sample, hidden_dim)
            
            # (hidden_dim)
            h_i_tensor = h[idx,:] # got 1 dim 
            # (1,hidden_dim)
            h_i_tensor = h_i_tensor[None,:] # got 2 dim 
            
            # preparing to broadcast up to # repeated labels
            h_i_tensor = h_i_tensor.repeat(np.count_nonzero(mask),1)

            
            #print("h_j idx :",h[mask,:,:].shape)
            # (seq_len,hidden_dim) , (#pairs, hidden_dim)
            if debug:
                if np.count_nonzero(mask) >= 2:
                    print("----")
                    print("masking label debug :",np.array(labels)[mask])
                    print("current labels ",np.array(labels)[idx])
                    print("---")
                
                print(">>>>>>>>>>>>>")
                print("repeat for broadcast :",h_i_tensor.shape)
                print("before append h_i and h_j")
                print("h_i : ",h_i_tensor.shape)
                print("h_j : ",h[mask,:].shape)


            # proof masking are correct they select the same class 
            h_i.append(h_i_tensor)
            h_j.append(h[mask,:])


            for val in idxs_arr[mask]: 
                skips.append(val)
            # add pair numbers of samples 

            # copy sample i to #of pairs
            for i in range(np.count_nonzero(mask)):
                idx_i.append(idx)

            T+= np.count_nonzero(mask)
            
            if debug:
                
                print("idx:",idx)
                print("current skips :",idxs_arr[mask])
                print("current labels :",label)

                label_arr = np.array(labels)

                 
                print("pair class :",label_arr[mask])
                print("mask:", mask)
                print("count:",len(mask))
                print("numbers of pairs one label :",np.count_nonzero(mask))
           
    
    if h_i:
    # after ending loop 
        h_i = torch.cat(h_i,dim=0)
        h_j = torch.cat(h_j,dim=0)    
    
        
        """
        print("all the sample i :",idx_i)
        print("skips :",skips)
        print("h_i shape :",h_i.shape)
        print("the number of pairs :",T)
        """
    
        if debug: 

            print("the number of pairs for entire batch:",T) 
            print("pairs see from labels : ",len(labels)-len(set(labels)))
    


        return T, h_i, h_j, idx_i
    else:

        return T, None, None, None
    
    

def supervised_contrasive_loss(h_i:Tensor,h_j:Tensor,h_n:Tensor,T:int,temp,idx_yij:List,callback=None,debug=False)->Union[ndarray, Tensor]:
    """
    T - number of pairs from the same classes in batch
    
    pos_pair - two utterances from the same class
    
    * remark previous work treat the utterance and itself as pos_pair
    neg_pair - two utterances across different class  
   
    """
    sim = Similarity(temp)
    
    device = "cuda:2"


    if callback != None:
       
       h_i = callback(h_i)
       h_j = callback(h_j)
       h_n = callback(h_n)
           
    # exp(sim(a,b)/ temp)

    pos_sim = torch.exp(sim(h_i,h_j))
     
    # for collect compute  sum_batch(exp(sim(hi,hn)/t)) 
    bot_sim  = []

    # masking bottom
    # same batch size shape
    #mask = np.arange(h_n.shape[0])

    """
    print("idxes yi=yj :",idx_yij)
    print("len yij",len(idx_yij))
    print("mask :",mask)
    print("# hi:  ",h_i.shape[0])
    """
    for idx in range(h_i.shape[0]):
       

        #mask = mask != idx_yij[idx]

        #h_n_neg = h_n[mask,:]
        # create h_i equal h_n_neg.shape[0] copies

        # select current sample from list pos pairs
        h_i_broad = h_i[idx].repeat(h_n.shape[0],1)
        

        if debug:
            print("h_i before broad :",h_i[idx].shape)
            print("after broad h_i to h_n",h_i_broad.shape)
            print("h_n shape :",h_n.shape)
        

        # sum over batch
        res = torch.sum(torch.exp(sim(h_i_broad,h_n))) 
        
        if debug:
            print("sim(h_i,h_n) shape :",res.shape)
            print("neg_sim max_min :",res.max(), res.min())
         
   
        if debug:
            print("after summing bottom factor :",res.shape)


        # to use with each pair i and j 
        bot_sim.append(res)

    bot_sim = torch.Tensor(bot_sim).to(device)

    if debug:
        print("bot_sim :",bot_sim.shape)
        print("pos_sim.shape :",pos_sim.shape)     
    
       

    if debug:
        print("bot sim :",bot_sim.shape)
        print("pos sim :",pos_sim.shape)
    
    
    loss = torch.log((pos_sim/bot_sim))
    
    
    loss = torch.sum(loss)

    if debug:
        print("after take log: ",loss)
    
    loss = -loss / T   

    
    return loss
