from keras.engine.topology import Layer
import tensorflow as tf
import keras
from keras import backend as K
class PixelAttention(Layer):
    def __init__(self,kernel_range=[3,3],shift=1,ff_kernel=[3,3],useBN=False,**kwargs):
        self.kernel_range=kernel_range # should be a list
        self.shift=shift
        self.ff_kernel=ff_kernel
        self.useBN=useBN
        super(PixelAttention,self).__init__(**kwargs)

    def build(self,input_shape):
        D=input_shape[-1]
        self.K_P=self.add_weight(name='K_P',shape=(D,D),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.V_P=self.add_weight(name='V_P',shape=(D,D),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.Q_P=self.add_weight(name='Q_P',shape=(D,D),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.ff1_kernel=self.add_weight(name='ff1_kernel',
                                        shape=(3,3,D,D),
                                        initializer='glorot_uniform',trainable=True)
        self.ff1_bais=self.add_weight(name='ff1_bias',
                                        shape=(D,),initializer='glorot_uniform',trainable=True)
        self.ff2_kernel=self.add_weight(name='ff2_kernel',
                                        shape=(3,3,D,2*D),
                                        initializer='glorot_uniform',trainable=True)
        self.ff2_bais=self.add_weight(name='ff2_bias',
                                        shape=(2*D,),initializer='glorot_uniform',trainable=True)

        self.ff3_kernel=self.add_weight(name='ff3_kernel',
                                        shape=(3,3,2*D,D),
                                        initializer='glorot_uniform',trainable=True)
        self.ff3_bais=self.add_weight(name='ff3_bias',
                                        shape=(D,),initializer='glorot_uniform',trainable=True)
                                                                        
        super(PixelAttention,self).build(input_shape)
    
    def call(self,x):
        _,_,_,D=x.shape
        s=K.shape(x)
        h_half=self.kernel_range[0]//2
        w_half=self.kernel_range[1]//2

        ls=list()
        masks=list()
        mask_x=tf.ones(shape=(s[0],s[1],s[2],1))
        paddings=tf.constant([[0,0],[h_half*self.shift,h_half*self.shift],[w_half*self.shift,w_half*self.shift],[0,0]])
        xt=tf.pad(x,paddings,"CONSTANT")
        mask_pad=tf.pad(mask_x,paddings,"CONSTANT")
        c_x,c_y=h_half*self.shift,w_half*self.shift
        for i in range(-h_half,h_half+1):
            for j in range(-w_half,w_half+1):
                _t=xt[:,c_x+i*self.shift:c_x+i*self.shift+s[1],c_y+j*self.shift:c_y+j*self.shift+s[2],:]
                _m=mask_pad[:,c_x+i*self.shift:c_x+i*self.shift+s[1],c_y+j*self.shift:c_y+j*self.shift+s[2],:]
                ls.append(_t)
                masks.append(_m)

        stk=tf.stack(ls,axis=3,name="stack")
        m_stack=tf.stack(masks,axis=3,name="mask")
        m_vec=tf.reshape(m_stack,shape=[s[0]*s[1]*s[2],self.kernel_range[0]*self.kernel_range[1],1])
        q_original=tf.reshape(x,shape=[s[0]*s[1]*s[2],D])
        q_vector=tf.matmul(q_original,self.Q_P)
        q=tf.reshape(q_vector,shape=[s[0]*s[1]*s[2],1,D])
        kv_original=tf.reshape(stk,shape=[s[0]*s[1]*s[2]*self.kernel_range[0]*self.kernel_range[1],D])
        k_vector=tf.matmul(kv_original,self.K_P)
        k=tf.reshape(k_vector,shape=[s[0]*s[1]*s[2],self.kernel_range[0]*self.kernel_range[1],D])
        v_vector=tf.matmul(kv_original,self.V_P)
        v=tf.reshape(v_vector,shape=[s[0]*s[1]*s[2],self.kernel_range[0]*self.kernel_range[1],D])
        alpha=tf.nn.softmax(tf.matmul(k,q,transpose_b=True)*m_vec,axis=1) 
        __res=tf.matmul(alpha,v,transpose_a=True)
        _res=tf.reshape(__res,shape=[s[0],s[1],s[2],D])

        t=x+_res
        if self.useBN:
            t=keras.layers.BatchNormalization(axis=-1)(t)
        _t=t
        t=tf.nn.relu(tf.nn.conv2d(input=t,filter=self.ff1_kernel,padding='SAME',strides=[1,1,1,1])+self.ff1_bais)
        t=tf.nn.relu(tf.nn.conv2d(input=t,filter=self.ff2_kernel,padding='SAME',strides=[1,1,1,1])+self.ff2_bais)
        t=tf.nn.relu(tf.nn.conv2d(input=t,filter=self.ff3_kernel,padding='SAME',strides=[1,1,1,1])+self.ff3_bais)
        t=_t+t
        if self.useBN:
            res=keras.layers.BatchNormalization(axis=-1)(t)
        else:
            res=t
        return res



    def compute_output_shape(self,input_shape):
        return input_shape