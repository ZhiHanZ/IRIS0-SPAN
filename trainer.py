import os
import keras
import numpy as np
np.set_printoptions( 3, suppress = True )
from tensorflow.python.client import device_lib
from datasets import CASIA, NIST, coverage, COLUMBIA
import json
import cv2

freeze_featex = True
pretrain = True
config_filename = "configs/config_CASIA_RESIZE_02.json"
weight_file = "PixelAttention32.h5"
debug = False
nb_gpus = 2
batch_size = 1

with open(config_filename) as obj:
    json_str=obj.read()
json_setting=json.loads(json_str)
base_model_idx =  json_setting["Project_Setting"]["baselineIndex"]

#################################################################################
# Set experiment parameters
#################################################################################
model_name = "PixelAttention{}".format( base_model_idx )
expt_root = "PixelAttention/{}".format( model_name )
os.system( 'mkdir -p {}'.format( expt_root ) )

if debug :
    nb_train_batches_per_epoch = 10
    nb_valid_batches_per_epoch = 5
else :
    nb_train_batches_per_epoch = 1000
    nb_valid_batches_per_epoch = 500

# # engine_bsize = [1] * len(training_dataset_list)


# prepare data generator
print ("INFO: use batch_size =", batch_size)
from utils.data_utils import dataGeneratorGateway
train_datagen, valid_datagen = dataGeneratorGateway(finetuneDataLoader= coverage.CoverageLoader(),pretrain=pretrain, batch_size=batch_size, debug=debug, test_size=0.2)

#################################################################################
# Set Model
#################################################################################
from models import ManTraNetv3 as mm

Pro=mm.ManTraNet(config_filename)
#model=Pro.get_model_0210()
#model = Pro.get_model_1010()
model = Pro.get_model_1010_resize()
#model = Pro.get_model_0301()
#model = Pro.get_model_03_04()

from utils.utils_mask_only_v3 import prepare_callbacks, rec, pre, F1
from utils import utils_mask_only_v3
from imp import reload
reload(utils_mask_only_v3)


my_callbacks = utils_mask_only_v3.prepare_callbacks(expt_root, model_name, time_limit='333:59:59')
para_model = keras.utils.multi_gpu_model( model, nb_gpus )
init_weight = weight_file
init_lr = 1e-4
init_epoch = 0
para_model.load_weights(weight_file)

from keras.optimizers import Adam
optimizer = Adam(init_lr)

from utils.metrics import auroc

para_model.compile( optimizer=optimizer,
                    loss = 'binary_crossentropy',
                    metrics = [F1, rec, pre, auroc] )

para_model.fit_generator( train_datagen,
                              epochs=500 if not debug else 1,
                              verbose=1,
                              workers=4,
                              initial_epoch=init_epoch,
                              max_queue_size=16,
                              callbacks = my_callbacks,
                              validation_data=valid_datagen,
                              )


