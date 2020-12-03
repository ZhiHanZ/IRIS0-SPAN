#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
from matplotlib.image import imread,imsave
import os
from models import modelCore

from keras.layers import Input, ConvLSTM2D, BatchNormalization
from keras.constraints import unit_norm
from keras.models import Model
import cv2
from datetime import datetime 
import random
from utils.metrics import np_F1
from utils.metrics import np_ap
from utils.metrics import np_auc
from utils.metrics import np_F1_k
from layers import Encoder, Decoder, PixelAttention as pa, MultiHeadPixelAttention as mpa
from utils.image_utils import read_rgb_image, decode_an_image_file, postprocess
from layers.MyLayers import *
from datasets.coverage import prepare_coverage_dataset


class ManTraNet:
#member variables
    setting=None
    bs=None
    
    manTraNet_root=None
    manTraNet_srcDir=None
    manTraNet_modelDir=None
    manTraNet_dataDir=None
    
    pretrain_index=4
    weight_file=None
    window_size_list=[7,15,31]
    IMC_model_idx=2
    
    #For private test
    sample_file=None
    baselinename=None
#member functions
    def __init__(self,config_fileName):
        json_str=None
        with open(config_fileName) as obj:
            json_str=obj.read()
        self.setting=json.loads(json_str)["Project_Setting"]
        self.manTraNet_root=self.setting["manTraNet_root"]
        self.manTraNet_srcDir=os.path.join(self.manTraNet_root,self.setting["srcDir"])
        self.manTraNet_modelDir=os.path.join(self.manTraNet_root,self.setting["modelDir"])
        self.manTraNet_dataDir=os.path.join(self.manTraNet_root,self.setting["data_dir"])
        self.weight_file="{}/ManTraNet_Ptrain{}.h5".format(self.manTraNet_modelDir,self.pretrain_index)
        
        self.sample_file = os.path.join(self.manTraNet_dataDir, 'samplePairs.csv' )
        #self.bs=baseline_funcs("config.json")
        self.baselinename=self.setting["baselinename"]
        
        self.encoder= Encoder.Encoder(config_fileName)
        self.decoder= Decoder.Decoder(config_fileName)
        
    def get_Featex(self,trainable=False):
        type_idx = self.IMC_model_idx if self.IMC_model_idx < 4 else 2
        Featex = modelCore.create_featex_vgg16_base(type_idx)
        Featex.trainable=trainable
        return Featex
    def get_original_model_0(self):
        model= modelCore.load_pretrain_model_by_index(0, self.manTraNet_modelDir)
        return model
    def get_original_model_by_id(self, modelId):
        assert modelId >=0 and modelId < 5, "only 0 - 4 acceptable"
        model= modelCore.load_pretrain_model_by_index(modelId, self.manTraNet_modelDir)
        return model

    def get_original_model_4(self):
        model= modelCore.load_pretrain_model_by_index(4, self.manTraNet_modelDir)
        return model

    def get_original_model(self):
        img_in = Input(shape=(None,None,3), name='img_in')
        Featex=self.get_Featex()
        rf=Featex(img_in)
        rf = Conv2D( 64, (1,1), activation=None, use_bias=False, kernel_constraint = unit_norm( axis=-2 ),
                 name='outlierTrans_t', padding = 'same' )(rf)
        bf = BatchNormalization( axis=-1, name='bnorm_t', center=False, scale=False )(rf)
        devf5d = modelCore.NestedWindowAverageFeatExtrator(window_size_list=self.window_size_list, include_global=False,  # not add global, it's just a test!!!
                                                           output_mode='5d', minus_original=True, name='nestedAvgFeatex_t')( bf )
        apply_normalization=True
        if ( apply_normalization ) :
            sigma = modelCore.GlobalStd2D(name='glbStd_t')(bf)
            sigma5d = Lambda( lambda t : K.expand_dims( t, axis=1 ), name='expTime_t')( sigma )
            devf5d = Lambda( lambda vs : K.abs(vs[0]/vs[1]), name='divStd_t' )([devf5d, sigma5d])
        
        devf = ConvLSTM2D( 8, (7,7),
                       activation='tanh',
                       recurrent_activation='hard_sigmoid',
                       padding='same',
                       name='cLSTM_t',
                       return_sequences=False )(devf5d)
        pred_out = Conv2D(1, (7,7), padding='same', activation='sigmoid', name='pred_t')( devf )
        model=Model( inputs=img_in, outputs=pred_out, name='sigNet_t')
        model.load_weights(self.weight_file,by_name=True)
        return model
    
    def Last_Layer_0725(self,x):
        t=Conv2D( 32, (5,5), activation='relu', name='final_1', padding = 'same' )(x)
        t=Conv2D( 16, (5,5), activation='relu', name='final_2', padding = 'same' )(t)
        t=Conv2D( 8, (5,5), activation='relu', name='final_3', padding = 'same' )(t)
        t=Conv2D( 4, (5,5), activation='relu', name='final_4', padding = 'same' )(t)
        t=Conv2D( 1, (5,5), activation='sigmoid', name='final_5', padding = 'same' )(t)
        return t

    def Last_Layer_0903(self,x):
        t=Conv2D( 16, (5,5), activation='relu', name='final_2', padding = 'same' )(x)
        t=Conv2D( 8, (5,5), activation='relu', name='final_3', padding = 'same' )(t)
        t=Conv2D( 4, (5,5), activation='relu', name='final_4', padding = 'same' )(t)
        t=Conv2D( 1, (5,5), activation='sigmoid', name='final_5', padding = 'same' )(t)
        return t

    def Last_Layer_0726(self,x):
        t=Conv2D( 128, (5,5), activation='relu', name='final_1', padding = 'same' )(x)
        t=Conv2D( 64, (5,5), activation='relu', name='final_2', padding = 'same' )(t)
        t=Conv2D( 32, (5,5), activation='relu', name='final_3', padding = 'same' )(t)
        t=Conv2D( 16, (5,5), activation='relu', name='final_4', padding = 'same' )(t)
        t=Conv2D( 1, (5,5), activation='sigmoid', name='final_5', padding = 'same' )(t)
        return t
    def get_model_1010(self,layers_steps=[1,3,9,27,81]):
        img_in = Input(shape=(None,None,3), name='img_in')
        Featex=self.get_Featex()
        rf=Featex(img_in)
        rf = Conv2D( 32, (1,1), activation=None, use_bias=False, kernel_constraint = unit_norm( axis=-2 ),
                 name='outlierTrans_new', padding = 'same' )(rf)
        t=rf
        for step in layers_steps:
            t=pa.PixelAttention(shift=step,useBN=False, useRes=True)(t)
            
        pred_out=self.Last_Layer_0725(t)
        model=Model( inputs=img_in, outputs=pred_out, name='sigNet')
        model.load_weights(self.weight_file,by_name=True)
        return model


    def get_model_1010_resize(self,layers_steps=[1,3,9,27,81]):
        img_in = Input(shape=(None,None,3), name='img_in')
        Featex=self.get_Featex()
        rf=Featex(img_in)
        resize = Lambda( lambda t : tf.image.resize_images(t, (224, 224)), name='resize')( rf ) 
        rf = Conv2D( 32, (1,1), activation=None, use_bias=False, kernel_constraint = unit_norm( axis=-2 ),
                 name='outlierTrans_new', padding = 'same' )(resize)
        t=rf
        for step in layers_steps:
            t=pa.PixelAttention(shift=step,useBN=False, useRes=True)(t)
            
        pred_out=self.Last_Layer_0725(t)
        model=Model( inputs=img_in, outputs=pred_out, name='sigNet')
        model.load_weights(self.weight_file,by_name=True)
        return model
    def get_model_1010_resize_visualization(self,layers_steps=[1,3,9,27,81], path="demo/"):
        img_in = Input(shape=(None,None,3), name='img_in')
        Featex=self.get_Featex()
        rf=Featex(img_in)
        resize = Lambda( lambda t : tf.image.resize_images(t, (224, 224)), name='resize')( rf ) 
        rf = Conv2D( 32, (1,1), activation=None, use_bias=False, kernel_constraint = unit_norm( axis=-2 ),
                 name='outlierTrans_new', padding = 'same' )(resize)
        t=rf
        for step in layers_steps:
            t=pa.PixelAttention(shift=step,useBN=False, useRes=True, useVisual=True, visualPath=path + "step_{}".format(step))(t)
            
        pred_out=self.Last_Layer_0725(t)
        model=Model( inputs=img_in, outputs=pred_out, name='sigNet')
        model.load_weights(self.weight_file,by_name=True)
        return model
    def get_model_1010_pe(self,layers_steps=[1,3,9,27,81]):
        from layers.PixelAttention_pe import PixelAttention
        img_in = Input(shape=(None,None,3), name='img_in')
        Featex=self.get_Featex()
        rf=Featex(img_in)
        resize = Lambda( lambda t : tf.image.resize_images(t, (224, 224)), name='resize')( rf ) 
        rf = Conv2D( 32, (1,1), activation=None, use_bias=False, kernel_constraint = unit_norm( axis=-2 ),
                 name='outlierTrans_new', padding = 'same' )(resize)
        t=rf
        for step in layers_steps:
            t=PixelAttention(shift=step,useBN=False, useRes=True)(t)
            
        pred_out=self.Last_Layer_0725(t)
        model=Model( inputs=img_in, outputs=pred_out, name='sigNet')
        model.load_weights(self.weight_file,by_name=True)
        return model

    def get_model_0106(self, layers_steps=[1,3,9,27,81]):
        img_in = Input(shape=(None,None,3), name='img_in')
        Featex=self.get_Featex()
        rf=Featex(img_in)
        rf = Conv2D( 32, (1,1), activation=None, use_bias=False, kernel_constraint = unit_norm( axis=-2 ),
                 name='outlierTrans_new', padding = 'same' )(rf)
        t = rf
        arr = []
        for step in layers_steps:
            t = pa.PixelAttention(shift=step, useBN=False)(t)
            arr.append(t)
        attention_pool_layer = Lambda( lambda t : tf.stack(t, axis=1), name='attention_pool')( arr )
        print(attention_pool_layer.shape)
        devf = ConvLSTM2D( 8, (7,7),
                       activation='tanh',
                       recurrent_activation='hard_sigmoid',
                       padding='same',
                       name='cLSTM_t', 
                       return_sequences=False )(attention_pool_layer)
        pred_out = Conv2D(1, (7,7), padding='same', activation='sigmoid', name='pred_t')( devf )
        print(pred_out.shape)
        model=Model( inputs=img_in, outputs=pred_out, name='sigNet')
        model.load_weights(self.weight_file,by_name=True)
        return model
    def get_model_0210(self, layers_steps=[1,3,9,27,81]):
        img_in = Input(shape=(None,None,3), name='img_in')
        Featex=self.get_Featex()
        rf=Featex(img_in)
        rf = Conv2D( 32, (1,1), activation=None, use_bias=False, kernel_constraint = unit_norm( axis=-2 ),
                 name='outlierTrans_new', padding = 'same' )(rf)
        t = rf
        arr = []
        for step in layers_steps:
            t = mpa.MultiHeadPixelAttention(shift=step, useBN=False)(t)
            arr.append(t)
        attention_pool_layer = Lambda( lambda t : tf.stack(t, axis=1), name='attention_pool')( arr )
        print(attention_pool_layer.shape)
        devf = ConvLSTM2D( 8, (7,7),
                       activation='tanh',
                       recurrent_activation='hard_sigmoid',
                       padding='same',
                       name='cLSTM_t', 
                       return_sequences=False )(attention_pool_layer)
        pred_out = Conv2D(1, (7,7), padding='same', activation='sigmoid', name='pred_t')( devf )
        print(pred_out.shape)
        model=Model( inputs=img_in, outputs=pred_out, name='sigNet')
        model.load_weights(self.weight_file,by_name=True)
        return model
    def get_model_0202(self, layers_steps=[1,3,9,27,81]):
        img_in = Input(shape=(None,None,3), name='img_in')
        Featex=self.get_Featex()
        rf=Featex(img_in)
        rf = Conv2D( 32, (1,1), activation=None, use_bias=False, kernel_constraint = unit_norm( axis=-2 ),
                 name='outlierTrans_new', padding = 'same' )(rf)
        t = rf
        arr = []
        for step in layers_steps:
            t = pa.PixelAttention(shift=step, useBN=False)(t)
            arr.append(t)
        attention_pool_layer = Lambda( lambda t : tf.stack(t, axis=1), name='attention_pool')( arr )
        print(attention_pool_layer.shape)
        devf = ConvLSTM2D( 8, (7,7),
                       activation='tanh',
                       recurrent_activation='hard_sigmoid',
                       padding='same',
                       name='cLSTM_t', 
                       return_sequences=False )(attention_pool_layer)
        pred_out = Conv2D(1, (7,7), padding='same', activation='sigmoid', name='pred_t')( devf )
        print(pred_out.shape)
        model=Model( inputs=img_in, outputs=pred_out, name='sigNet')
        model.load_weights(self.weight_file,by_name=True)
        return model
    def get_model_0903(self): # 2 encoders and 2 decoders and 4 h••••••••••••••••••eaders
        img_in = Input(shape=(None,None,3), name='img_in')
        Featex=self.get_Featex()
        rf=Featex(img_in)
        rf = Conv2D( 32, (1,1), activation=None, use_bias=False, kernel_constraint = unit_norm( axis=-2 ),
                 name='outlierTrans_new', padding = 'same' )(rf)
        bf = BatchNormalization( axis=-1, name='bnorm_new', center=False, scale=False )(rf)
        devf5d = modelCore.NestedWindowAverageFeatExtrator(window_size_list=self.window_size_list,
                                                           output_mode='5d', minus_original=True, name='nestedAvgFeatex_new')( bf )
        apply_normalization=True
        if ( apply_normalization ) :
            sigma = modelCore.GlobalStd2D(name='glbStd_new')(bf)
            sigma5d = Lambda( lambda t : K.expand_dims( t, axis=1 ), name='expTime_new')( sigma )
            devf5d = Lambda( lambda vs : K.abs(vs[0]/vs[1]), name='divStd_new' )([devf5d, sigma5d])
        
        init_tensor=MyStartVariable()(devf5d)
        t=self.encoder.encoders(devf5d)
        t=self.decoder.decoders(init_tensor,t)
        t=Lambda(lambda t:t[:,0,...],name="lambda_out")(t)
        pred_out=self.Last_Layer_0903(t)
        model=Model( inputs=img_in, outputs=pred_out, name='sigNet')
        
        model.load_weights(self.weight_file,by_name=True)
        return model
    
    def get_model_0801(self):
        img_in = Input(shape=(None,None,3), name='img_in')
        Featex=self.get_Featex()
        rf=Featex(img_in)
        #t=Conv2D(128,kernel_size=(3,3),strides=(2,2),activation="relu",padding="same",name="step1_0801")(rf)
        t=Conv2D(128,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",name="step2_0801")(rf)
        #t=Conv2D(256,kernel_size=(3,3),strides=(2,2),activation="relu",padding="same",name="step3_0801")(t)
        t=Conv2D(256,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",name="step4_0801")(t)

        t=Conv2D(256,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",dilation_rate=2,name="step5_0801")(t)
        t=Conv2D(256,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",dilation_rate=4,name="step6_0801")(t)
        t=Conv2D(256,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",dilation_rate=8,name="step7_0801")(t)
        t=Conv2D(256,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",dilation_rate=16,name="step8_0801")(t)

        t=Conv2D(256,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",name="step9_0801")(t)
        t=Conv2D(256,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",name="step10_0801")(t)

        #t=keras.layers.Conv2DTranspose(128,kernel_size=4,strides=2,padding="same",name="step11_0801")(t)
        t=Conv2D(128,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",name="step12_0801")(t)
        #t=keras.layers.Conv2DTranspose(64,kernel_size=4,strides=2,padding="same",name="step13_0801")(t)
        t=Conv2D(32,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",name="step14_0801")(t)
        t=Conv2D(8,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",name="step15_0801")(t)
        t=Conv2D(1,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same",name="step16_0801")(t)
        
        model=Model( inputs=img_in, outputs=t, name='sigNet_0801')
        print (model.summary( line_length=120 ))
        model.load_weights(self.weight_file,by_name=True)
        return model

    def get_model_0726(self): #2 encoders and 8 headers
        img_in = Input(shape=(None,None,3), name='img_in')
        Featex=self.get_Featex()
        rf=Featex(img_in)
        rf = Conv2D( 64, (1,1), activation=None, use_bias=False, kernel_constraint = unit_norm( axis=-2 ),
                 name='outlierTrans_new', padding = 'same' )(rf)
        bf = BatchNormalization( axis=-1, name='bnorm_new', center=False, scale=False )(rf)
        devf5d = modelCore.NestedWindowAverageFeatExtrator(window_size_list=self.window_size_list,
                                                           output_mode='5d', minus_original=True, name='nestedAvgFeatex_new')( bf )
        apply_normalization=True
        if ( apply_normalization ) :
            sigma = modelCore.GlobalStd2D(name='glbStd_new')(bf)
            sigma5d = Lambda( lambda t : K.expand_dims( t, axis=1 ), name='expTime_new')( sigma )
            devf5d = Lambda( lambda vs : K.abs(vs[0]/vs[1]), name='divStd_new' )([devf5d, sigma5d])
        
        
        t=self.encoder.encoders(devf5d)
        
        def helper(x):
            s=tf.shape(x)
            sc=x.shape
            t=tf.transpose(x,(0,2,3,1,4))
            t=tf.reshape(x,shape=(s[0],s[2],s[3],sc[1]*sc[4]))
            return t
        t=Lambda(helper,name="lambda_out")(t)
        pred_out=self.Last_Layer_0726(t)

        
        model=Model( inputs=img_in, outputs=pred_out, name='sigNet')
        model.load_weights(self.weight_file,by_name=True)
        return model
