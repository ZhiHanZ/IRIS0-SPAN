#!/usr/bin/env python
# coding: utf-8

# In[1]:


from layers.MyLayers import *
import json


# In[2]:


class Decoder:
    setting=None
    n_heads=None
    n_decoder=None
    useBN=None
    ksize=None
    def __init__(self,config_fileName):
        json_str=None
        with open(config_fileName) as obj:
            json_str=obj.read()
        self.setting=json.loads(json_str)["Decoder_stack"]
        self.n_heads=self.setting["n_heads"]
        self.n_decoder=self.setting["n_decoder"]
        self.useBN=self.setting["useBN"] # not use
        self.ksize=self.setting["ksize"]
        
    def one_decoder(self,q,kv,ind):
        selfAtt=Multi_SelfAttention2D(qkv=q,ksize=self.ksize,n_heads=self.n_heads,name="decoder"+str(ind)+"_step1")
        norm1=Add_Normalize(selfAtt,q,name="decoder"+str(ind)+"_step2")
        att=Multi_Attention2D(n_heads=self.n_heads,ksize=self.ksize,name="decoder"+str(ind)+"_step3")([norm1,kv,kv])
        norm2=Add_Normalize(att,norm1,name="decoder"+str(ind)+"_step4")
        ff=Feed_Forward(hidden_layer_dims=[64,32],name="decoder"+str(ind)+"_step5")(norm2)
        norm3=Add_Normalize(ff,norm2,name="decoder"+str(ind)+"_step6")
        return norm3
    
    def decoders(self,x,kv):
        for i in range(self.n_decoder):
            x=self.one_decoder(x,kv,ind=i)
        return x


# In[3]:


if __name__=="__main__":
    t=Decoder("config.json")


# In[ ]:




