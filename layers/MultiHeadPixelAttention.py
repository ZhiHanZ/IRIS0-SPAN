from keras.engine.topology import Layer
import tensorflow as tf
import keras
from keras import backend as K
class MultiHeadPixelAttention(Layer):

    def __init__(self,kernel_range=[3,3],numHead =  2, shift=1,ff_kernel=[3,3],useBN=False, useRes=False,**kwargs):
        self.kernel_range=kernel_range # should be a list
        self.shift=shift
        self.numHead = numHead
        self.ff_kernel=ff_kernel
        self.useBN=useBN
        self.useRes=useRes
        super(MultiHeadPixelAttention,self).__init__(**kwargs)

    def build(self,input_shape):
        D=input_shape[-1]
        n_p=self.kernel_range[0]*self.kernel_range[1]
        self.K_P=self.add_weight(name='K_P',shape=(1,1,D,D*n_p),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.V_P=self.add_weight(name='V_P',shape=(1,1,D,D*n_p),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.Q_P=self.add_weight(name='Q_P',shape=(1,1,D,D),
                                    initializer='glorot_uniform',
                                    trainable=True)


        self.ff1_kernel=self.add_weight(name='ff1_kernel',
                                        shape=(1,1,D,D),
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
                                                                        
        super( MultiHeadPixelAttention,self).build(input_shape)
    
    def call(self,x):
        h_half=self.kernel_range[0]//2
        w_half=self.kernel_range[1]//2
        _,_,_,D=x.shape
        s=K.shape(x)
        x_k=tf.nn.conv2d(input=x,filter=self.K_P,padding="SAME",strides=[1,1,1,1])
        x_v=tf.nn.conv2d(input=x,filter=self.V_P,padding="SAME",strides=[1,1,1,1])
        x_q=tf.nn.conv2d(input=x,filter=self.Q_P,padding="SAME",strides=[1,1,1,1])
        paddings=tf.constant([[0,0],[h_half*self.shift,h_half*self.shift],[w_half*self.shift,w_half*self.shift],[0,0]])
        x_k=tf.pad(x_k,paddings,"CONSTANT")
        x_v=tf.pad(x_v,paddings,"CONSTANT")
        mask_x=tf.ones(shape=(s[0],s[1],s[2],self.numHead))
        mask_pad=tf.pad(mask_x,paddings,"CONSTANT")

        k_ls=list()
        v_ls=list()
        masks=list()
        
        c_x,c_y=h_half*self.shift,w_half*self.shift
        layer=0
        for i in range(-h_half,h_half+1):
            for j in range(-w_half,w_half+1):
                k_t=x_k[:,c_x+i*self.shift:c_x+i*self.shift+s[1],c_y+j*self.shift:c_y+j*self.shift+s[2],layer*D:(layer+1)*D]
                k_ls.append(k_t)

                v_t=x_v[:,c_x+i*self.shift:c_x+i*self.shift+s[1],c_y+j*self.shift:c_y+j*self.shift+s[2],layer*D:(layer+1)*D]
                v_ls.append(v_t)

                _m=mask_pad[:,c_x+i*self.shift:c_x+i*self.shift+s[1],c_y+j*self.shift:c_y+j*self.shift+s[2],:]
                masks.append(_m)
                layer+=1
        m_stack=tf.stack(masks,axis=3,name="mask") # B H M 9 4
        m_stack = tf.transpose(m_stack, perm=[0,1,2,4,3]) # B H M 4 9
        m_vec=tf.reshape(m_stack,shape=[self.numHead*s[0]*s[1]*s[2],self.kernel_range[0]*self.kernel_range[1],1])

        #m_list = [m_vec]*self.numHead
        #m_stack = tf.stack(m_list, axis=1, name="mask1")
        #m_vec=tf.reshape(m_stack,shape=[self.numHead*s[0]*s[1]*s[2],self.kernel_range[0]*self.kernel_range[1],1])
        
        
        k_stack=tf.stack(k_ls,axis=3,name="k_stack") # B H M 9, 32
        v_stack=tf.stack(v_ls,axis=3,name="v_stack") # B H M 9 32
        headDim = int(D//self.numHead)
        multiHeadK = tf.reshape(k_stack, (s[0],s[1],s[2], self.kernel_range[0]*self.kernel_range[1], self.numHead, headDim))
        multiHeadK = tf.transpose(multiHeadK, perm=[0,1,2,4,3,5]) # B * H * W * numHead * 9 * headDim
        multiHeadV = tf.reshape(v_stack ,(s[0],s[1],s[2], self.kernel_range[0]*self.kernel_range[1], self.numHead, headDim))
        multiHeadV = tf.transpose(multiHeadV, perm=[0,1,2,4,3,5]) # B * H * W * numHead * 9 * headDim
        multiHeadQ = tf.reshape(x_q, (s[0], s[1], s[2], 1, self.numHead, headDim))
        multiHeadQ = tf.transpose(multiHeadQ, perm=[0,1,2,4,3,5]) # B * H * W * numHead * 1 * headDim
        k=tf.reshape(multiHeadK,shape=[self.numHead*s[0]*s[1]*s[2],self.kernel_range[0]*self.kernel_range[1], headDim])
        v=tf.reshape(multiHeadV,shape=[self.numHead*s[0]*s[1]*s[2],self.kernel_range[0]*self.kernel_range[1], headDim])
        q=tf.reshape(multiHeadQ,shape=[self.numHead*s[0]*s[1]*s[2],1,headDim])
        alpha=tf.nn.softmax(tf.matmul(k,q,transpose_b=True)*m_vec/3,axis=1) 
        __res=tf.matmul(alpha,v,transpose_a=True)
        ___res = tf.reshape(__res, shape=[s[0],s[1],s[2], self.numHead, headDim, 1])
        _res=tf.reshape(___res,shape=[s[0],s[1],s[2],D])
        if self.useRes:
            t=x+_res
        else:
            t=_res
        if self.useBN:
            t=keras.layers.BatchNormalization(axis=-1)(t)
        _t=t
        t=tf.nn.relu(tf.nn.conv2d(input=t,filter=self.ff1_kernel,padding='SAME',strides=[1,1,1,1])+self.ff1_bais)
        t=tf.nn.relu(tf.nn.conv2d(input=t,filter=self.ff2_kernel,padding='SAME',strides=[1,1,1,1])+self.ff2_bais)
        t=tf.nn.relu(tf.nn.conv2d(input=t,filter=self.ff3_kernel,padding='SAME',strides=[1,1,1,1])+self.ff3_bais)
        if self.useRes:
            t=_t+t
        if self.useBN:
            res=keras.layers.BatchNormalization(axis=-1)(t)
        else:
            res=t
        return res



    def compute_output_shape(self,input_shape):
        return input_shape
