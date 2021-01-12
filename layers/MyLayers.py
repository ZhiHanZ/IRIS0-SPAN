#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Layer
from keras import backend as K
import keras
import tensorflow as tf
import json
from keras.layers import Lambda
from keras.layers import Conv2D


# In[ ]:


class Multi_Attention2D(Layer):
    def __init__(self,ksize,n_heads,
                 **kwargs):
        
        self.ksize=ksize
        self.n_heads=n_heads
        super(Multi_Attention2D,self).__init__(**kwargs)
     
    def build(self,input_shape):
        
        #print("input:",input_shape)
        if not isinstance(input_shape, list):
            raise ValueError('Multi_Attention2D layer should be called on a list of inputs.')
        if len(input_shape) !=3:
            raise ValueError('Multi_Attention2D should have 3 inputs as q,k,v')
        s_q=input_shape[0]
        s_k=input_shape[1]
        s_v=input_shape[2]
        self.res_s_q=s_q
        self.res_s_k=s_k
        self.res_s_v=s_v
        self.output_dim=s_v[4]
        q_dim=s_q[-1]
        k_dim=s_k[-1]
        v_dim=s_v[-1]
        q_kdim=q_dim*2//self.n_heads
        k_kdim=k_dim*2//self.n_heads
        v_kdim=v_dim*2//self.n_heads
        q_kernels_dim=q_kdim*self.n_heads
        k_kernels_dim=k_kdim*self.n_heads
        v_kernels_dim=v_kdim*self.n_heads 
        self.q_kernels=self.add_weight(name='q_kernels',
                                       shape=(self.ksize,self.ksize,q_dim,q_kernels_dim),
                                       initializer='glorot_uniform',trainable=True)
        #self.q_kernels_bias=self.add_weight(name='q_kernels_bias', shape=((q_kernels_dim,)),
        #                               initializer='glorot_uniform',trainable=True)
        
        self.k_kernels=self.add_weight(name='k_kernels',
                                       shape=(self.ksize,self.ksize,k_dim,k_kernels_dim),
                                       initializer='glorot_uniform',trainable=True)
        #self.k_kernels_bias=self.add_weight(name='k_kernels_bias', shape=((k_kernels_dim,)),
        #                               initializer='glorot_uniform',trainable=True)
        
        self.v_kernels=self.add_weight(name='v_kernels',
                                       shape=(self.ksize,self.ksize,v_dim,v_kernels_dim),
                                       initializer='glorot_uniform',trainable=True)
        #self.v_kernels_bias=self.add_weight(name='v_kernels_bias', shape=((v_kernels_dim,)),
        #                               initializer='glorot_uniform',trainable=True)
        
        self.out_kernels=self.add_weight(name='out_kernels',
                                         shape=(self.ksize,self.ksize,v_kernels_dim,self.output_dim),
                                        initializer='glorot_uniform',trainable=True)
        #self.out_kernels_bias=self.add_weight(name='out_kernels_bias', shape=((self.output_dim,)),
        #                               initializer='glorot_uniform',trainable=True)
        
        super(Multi_Attention2D, self).build(input_shape) 
        
    def call(self,x):
        q_tensor,k_tensor,v_tensor=x
        self.s_q=K.shape(q_tensor)
        self.s_k=K.shape(k_tensor)
        self.s_v=K.shape(v_tensor)
        self.q_dim=self.s_q[-1]
        self.k_dim=self.s_k[-1]
        self.v_dim=self.s_v[-1]
        self.q_kdim=self.q_dim*2//self.n_heads
        self.k_kdim=self.k_dim*2//self.n_heads
        self.v_kdim=self.v_dim*2//self.n_heads
        self.q_kernels_dim=self.q_kdim*self.n_heads
        self.k_kernels_dim=self.k_kdim*self.n_heads
        self.v_kernels_dim=self.v_kdim*self.n_heads 
        
        q_prepare=tf.reshape(q_tensor,shape=(self.s_q[0]*self.s_q[1],self.s_q[2],self.s_q[3],self.s_q[4])) # shape of (BS*q_T,H,W,q_dim)
        k_prepare=tf.reshape(k_tensor,shape=(self.s_k[0]*self.s_k[1],self.s_k[2],self.s_k[3],self.s_k[4])) # shape of (BS*kv_T,H,W,k_dim)
        v_prepare=tf.reshape(v_tensor,shape=(self.s_v[0]*self.s_v[1],self.s_v[2],self.s_v[3],self.s_v[4])) # shape of (BS*kv_T,H,W,v_dim)
        q=tf.nn.conv2d(input=q_prepare,filter=self.q_kernels,padding='SAME',strides=[1,1,1,1])#+self.q_kernels_bias # shape of (BS*q_T,H,W,q_kernels_dim)
        k=tf.nn.conv2d(input=k_prepare,filter=self.k_kernels,padding='SAME',strides=[1,1,1,1])#+self.k_kernels_bias # shape of (BS*kv_T,H,W,k_kernels_dim)
        v=tf.nn.conv2d(input=v_prepare,filter=self.v_kernels,padding='SAME',strides=[1,1,1,1])#+self.v_kernels_bias # shape of (BS*kv_T,H,W,v_kernels_dim)
        q_relu=q#tf.nn.relu(q)
        k_relu=k#tf.nn.relu(k)
        v_relu=v#tf.nn.relu(v)
        q_head_stacks=tf.split(q_relu,self.n_heads,axis=-1)
        q_restack=tf.reshape(tf.stack(q_head_stacks,axis=0),shape=(self.n_heads*self.s_q[0],self.s_q[1],self.s_q[2]*self.s_q[3]*self.q_kdim)) 
        k_head_stacks=tf.split(k_relu,self.n_heads,axis=-1)
        k_restack=tf.reshape(tf.stack(k_head_stacks,axis=0),shape=(self.n_heads*self.s_k[0],self.s_k[1],self.s_k[2]*self.s_k[3]*self.k_kdim)) 
        v_head_stacks=tf.split(v_relu,self.n_heads,axis=-1)
        v_restack=tf.reshape(tf.stack(v_head_stacks,axis=0),shape=(self.n_heads*self.s_v[0],self.s_v[1],self.s_v[2]*self.s_v[3]*self.v_kdim)) 
        scores=tf.nn.softmax(tf.matmul(q_restack,tf.transpose(k_restack,(0,2,1)))/8,axis=-1) # TO MODIFY!
        z_t=tf.reshape(tf.matmul(scores,v_restack),shape=(self.n_heads,self.s_v[0],self.s_q[1],self.s_v[2]*self.s_v[3]*self.v_kdim))
        z=tf.reshape(tf.transpose(z_t,(1,2,3,0)),shape=(self.s_v[0]*self.s_q[1],self.s_v[2],self.s_v[3],self.v_kernels_dim)) 
        
        out_t=tf.nn.conv2d(input=z,filter=self.out_kernels,padding='SAME',strides=[1,1,1,1])#+self.out_kernels_bias
        # out=K.reshape(tf.nn.relu(out_t),shape=(self.s_v[0],self.s_q[1],self.s_v[2],self.s_v[3],self.s_v[4]))
        out=K.reshape(out_t,shape=(self.s_v[0],self.s_q[1],self.s_v[2],self.s_v[3],self.s_v[4]))
        return out
    
    def compute_output_shape(self,input_shape):
        #print(input_shape)
        return (self.res_s_q[0],self.res_s_q[1],self.res_s_q[2],self.res_s_q[3],self.output_dim)


# In[ ]:


class Feed_Forward(Layer):
    def __init__(self,hidden_layer_dims,ksize=3,**kwargs):
        self.hidden_layer_dims=hidden_layer_dims
        assert isinstance(hidden_layer_dims,list)
        self.ksize=ksize
        super(Feed_Forward,self).__init__(**kwargs)
        
    def build(self,input_shape):
        self.w=list()
        self.b=list()
        prev_dim=input_shape[-1]
        for i in range(len(self.hidden_layer_dims)):
            self.w.append(self.add_weight(name='kernel'+str(i),
                                               shape=(self.ksize,self.ksize,prev_dim,self.hidden_layer_dims[i]),
                                               initializer='glorot_uniform',
                                               trainable=True))
            self.b.append(self.add_weight(name='bias'+str(i),
                                               shape=((self.hidden_layer_dims[i],)),
                                               initializer='glorot_uniform',
                                               trainable=True))
            prev_dim=self.hidden_layer_dims[i]
        super(Feed_Forward, self).build(input_shape)     
        
    def call(self,x):
        s=tf.shape(x)
        x=tf.reshape(x,shape=(s[0]*s[1],s[2],s[3],s[4]))
        for i in range(len(self.hidden_layer_dims)):
            x=tf.nn.conv2d(input=x,filter=self.w[i],strides=[1,1,1,1],padding="SAME")+self.b[i]
            x=tf.nn.relu(x)
        out=tf.reshape(x,shape=(s[0],s[1],s[2],s[3],s[4]))
        return out
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3],self.hidden_layer_dims[-1])


# In[ ]:


def Multi_SelfAttention2D(qkv,n_heads,ksize,name):
    return Multi_Attention2D(n_heads=n_heads,ksize=ksize,name=name)([qkv,qkv,qkv])

def Add_Normalize(x,y,name):
    z=keras.layers.Add()([x,y])
    #return tf.contrib.layers.layer_norm(z,begin_norm_axis=2,begin_params_axis=-1)
    # z=Lambda(lambda t:tf.cast(t,tf.float32))(z)
    z=keras.layers.BatchNormalization(axis=-1)(z)
    # z=Lambda(lambda t:tf.cast(t,tf.float16))(z)
    return z

