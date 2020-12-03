import json
import os

import cv2
import numpy as np
import keras
from parse import parse
from datetime import datetime
from utils.image_utils import RGB_normalization
from sklearn.model_selection import train_test_split
from datasets.dataset_interface import DataSetInterface
def parse_dataset_settings( dataset_full_name ) :
    dataset_name, resize_type = parse('{}-{}', dataset_full_name )
    return dataset_name, resize_type
def getData(testfile, paired_results, base_path="../"):
    with open(testfile, "r") as f:
        content = f.readlines()
        ans = [x.strip().split()[0] for x in content]
        ans = [base_path + x for x in ans]
        paired_train = [x for x in paired_results if x[0] in ans]
        paired_test = [x for x in paired_results if x[0] not in ans]
        return paired_train, paired_test

class DataGenerator(keras.utils.Sequence):
    def __init__(self, paired_results, preprocess, batch_size = 32,target_size=(224,224), shuffle=True, random_seed=12345, mode='training', reverse=False):
        self.batch_size = batch_size
        self.paired_results = paired_results
        self.shuffle=shuffle
        self.target_size = target_size
        self.on_epoch_end()
        self.prng = self._get_prng( random_seed )
        self.mode = mode
        self.preprocess = preprocess
        self.reverse = reverse
    def _get_prng( self, random_seed ) :
        return np.random.RandomState( random_seed )
    #will be called at the end of on epoch which shuffle the whole dataset
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paired_results))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    #length of the generator is the num of paired results over a batch size
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_results) / self.batch_size))
    #get images and their corresponding mask with size equal to batch size
    def _clip_image(self, raw, mask):
        if (raw.shape[0] == mask.shape[0] or raw.shape[1] == mask.shape[1]):
            raw = raw[:min(raw.shape[0], mask.shape[0]), :min(raw.shape[1], mask.shape[1]), :]
            mask = mask[:min(raw.shape[0], mask.shape[0]), :min(raw.shape[1], mask.shape[1])]
            return raw, mask
        else:
            return raw, mask

    def _get_batch(self, index):
        if ( self.mode == 'training' ) :
            batch_prng = self.prng
        else :
            batch_prng = self._get_prng( index )
        th, tw = self.target_size
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        paired_result_in_curr_batch = [self.paired_results[k] for k in indexes]
        imgs, masks = [], []
        for raw_file, mask_file in paired_result_in_curr_batch :
            r = cv2.imread( raw_file, 1 )[...,::-1]
            m = cv2.imread( mask_file, 0)
            #r, m = self._clip_image(r,m)
            h, w = r.shape[:2]
            #print(r.shape)
            rate = 512./min(h,w)
            nh, nw = int(h*rate), int(w*rate)
            r, m = self._random_resize( r, m, batch_prng, nh, nw )

            imgs.append(r)
            masks.append(m)
        return imgs, masks

    def _random_resize( self, image, mask, prng, th, tw ) :
        method = prng.choice( [ cv2.INTER_AREA,
                                cv2.INTER_CUBIC,
                                cv2.INTER_LANCZOS4,
                                cv2.INTER_LINEAR,
                                cv2.INTER_NEAREST ] )
        #new_image = cv2.resize( image, (tw, th), interpolation=method )
        new_image = image
        new_mask  = cv2.resize( mask,  (224, 224), interpolation=cv2.INTER_NEAREST )
        return new_image, new_mask
    def _preprocess_input( self, Z ) :
        return 2. * np.stack(Z,axis=0) / 255. - 1 # normalization
    def _preprocess_output( self, M ) :
        if self.reverse:
            normalizedM = [1 - img/255. for img in M]
        else:
            normalizedM = [img/255. for img in M]
        return np.expand_dims(np.stack(normalizedM,axis=0), axis=-1)
    def __getitem__(self, index):
        Z_list, M_list = self._get_batch( index )
        net_inputs = self._preprocess_input( Z_list )
        net_outputs = self._preprocess_output( M_list )
        # net_inputs, net_outputs = self.preprocess(Z_list, M_list) #di
        return net_inputs, net_outputs

# return train and validation generator
def finetuneDataBoot(finetuneDataLoader: DataSetInterface  = None, test_size = 0.2, random_state=1024, batch_size = 3):
    valid_dataset_names = {"CASIA", "NIST", "COVERAGE"}
    if finetuneDataLoader is None or finetuneDataLoader.get_name() not in valid_dataset_names:
        raise ValueError("not a valid dataloader for fine tune")
    reverse = True if finetuneDataLoader.get_name() == "NIST" else False
    if finetuneDataLoader.get_name() == "CASIA":
        trainingSet, validatingSet, lens, preprocess = finetuneDataLoader.load_trainset()
        print("INFO: num of training :", len(trainingSet), " num of validating: ", len(validatingSet))
    else :
        dataset, lens, preprocess = finetuneDataLoader.load_trainset()
        trainingSet, validatingSet = train_test_split(dataset, test_size=test_size, random_state=random_state)
        print("INFO: num of training :", len(trainingSet), " num of validating: ", len(validatingSet))
    train_datagen = DataGenerator(trainingSet, preprocess=preprocess, mode='training', batch_size=batch_size, reverse=reverse)
    validation_datagen = DataGenerator(validatingSet, preprocess=preprocess, mode='validation',
                                       batch_size=batch_size, reverse=reverse)
    return train_datagen, validation_datagen

def dataGeneratorGateway(finetuneDataLoader : DataSetInterface = None, test_size = 0.2, random_state=1024,
                         pretrain = True, debug = True, batch_size = 3,
                         use_7dataset=True, use_random_resize=True, use_random_compress=True,
                         force = False, use_tmproot = False,
                         ):
        return finetuneDataBoot(finetuneDataLoader = finetuneDataLoader, test_size=test_size, random_state=random_state, batch_size=batch_size)
