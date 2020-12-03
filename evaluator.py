from models import ManTraNetv3 as mm
import keras
import os
from datasets.CASIA import CASIADataLoader
from datasets.coverage import CoverageLoader
from datasets.NIST import NISTLoader
from utils.pipeline_utils import test_model
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
pretrain = False
config_filename= "configs/config_CASIA_RESIZE_02.json"
#config_filename= "configs/pretrain_configuration.json"
##look up the folder 
#weight_file = "PixelAttention/PixelAttention30/PixelAttention30_E0035-0.88481-0.0001000.h5"
#weight_file = "PixelAttention/PixelAttention30/PixelAttention30_E0040-0.88435-0.0001000.h5"
#weight_file = "PixelAttention/PixelAttention30/PixelAttention30_E0010-0.87065-0.0001000.h5"
#weight_file = "transformer16-CLSTM-7D-FF1-RR1-RC1-P63.h5"
#weight_file = "weights/PixelAttention34.h5"
#weight_file = "PixelAttention/PixelAttention3001/PixelAttention3001.h5"
#weight_file = "PixelAttention/PixelAttention5/PixelAttention5.h5"
#weight_file = "PixelAttention/PixelAttention11010/PixelAttention11010.h5"
weight_file = "PixelAttention32.h5"
##weight_file = "PixelAttention/PixelAttention39/PixelAttention39.h5"
#weight_file = "PixelAttention/PixelAttention30/PixelAttention30_E0010-0.87065-0.0001000.h5"
#weight_file = "PixelAttention/PixelAttention30/PixelAttention30_E0022-0.86994-0.0001000.h5"
#weight_file = "best_pretrain.h5"

Pro =mm.ManTraNet(config_filename)
#model = Pro.get_original_model_by_id(4)

## look up at target config16 description
#model = Pro.get_model_0106()
#model = Pro.get_model_0224()
model = Pro.get_model_1010_resize()
#model = Pro.get_model_1010_pe()
model = keras.utils.multi_gpu_model( model, 2)
    #Pro.test_model_on_coverage(model)
    #Pro.test_model_on_columbia(model)
model.load_weights(weight_file)
test_model(model, CoverageLoader(), reverseNorm=False, pretrain=pretrain)
