#!/usr/bin/env python
# coding: utf-8

# In[3]:


from layers.MyLayers import *
import json


# In[4]:


class Encoder:
    setting=None
    n_heads=None
    n_encoder=None
    useBN=None
    ksize=None
    def __init__(self,config_fileName):
        json_str=None
        with open(config_fileName) as obj:
            json_str=obj.read()
        self.setting=json.loads(json_str)["Encoder_stack"]
        self.n_heads=self.setting["n_heads"]
        self.n_encoder=self.setting["n_encoder"]
        self.useBN=self.setting["useBN"] # not use
        self.ksize=self.setting["ksize"]
        
    def one_encoder(self,inpt,ind):
        selfAtt=Multi_SelfAttention2D(qkv=inpt,ksize=self.ksize,n_heads=self.n_heads,name="encoder"+str(ind)+"_step1")
        norm1=Add_Normalize(selfAtt,inpt,name="encoder"+str(ind)+"_step2")
        ff=Feed_Forward(hidden_layer_dims=[64,32],name="encoder"+str(ind)+"_step3")(norm1)
        norm2=Add_Normalize(ff,norm1,name="encoder"+str(ind)+"_step4")
        return norm2

    def encoders(self,x):
        for i in range(self.n_encoder):
            x=self.one_encoder(x,ind=i)
        return x


# In[6]:


if __name__=="__main__":
    t=Encoder("config.json")


# In[ ]:




