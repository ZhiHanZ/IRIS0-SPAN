import os
import sys

import cv2
from matplotlib.image import imread,imsave
from utils.image_utils import read_rgb_image, decode_an_image_file, decode_an_image_array, postprocess
from utils.metrics import np_auc, np_ap, np_F1, np_F1_k
from utils.image_utils import RGB_normalization
from datasets.dataset_interface import DataSetInterface
from utils.metrics import compute_all
import tensorflow as tf
def test_model(model, loader : DataSetInterface, visual_path = None, log_path= None, raw=True, reverseNorm=False, pretrain=False):  # used for coverage dataset checking
    if not log_path is None:
        sys.stdout = Logger(log_path)
    if pretrain:
        input_pairs, _, preprocess = loader.load_testset()
    else:
        input_pairs, _, preprocess = loader.load_data()
    if visual_path is not None and not os.path.exists(visual_path):
        os.makedirs(visual_path)
    L = len(input_pairs)
    y_true = []
    y_pred = []
    print(loader.get_name())
    print("the number of test set is ", L)
    for ind in range(L):
        original_file, forged_file = input_pairs[ind]
        original = read_rgb_image(original_file)
        forged = cv2.imread(forged_file, 0) if loader.get_name() != "COLUMBIA" else cv2.imread( forged_file, 1 )[...,1]
        original, forged = preprocess(original, forged)
        _, mask, ptime = decode_an_image_file(original, model)
        mask = postprocess(mask, forged.shape)
        if visual_path is not None:
            visualization(visual_path, ind, original, forged, mask*255., raw=raw)
        y_true.append(forged)
        y_pred.append(mask)
    with tf.compat.v1.Session():
        print(compute_all(y_true, y_pred))
def visualization(path, ind, original, forged, mask, raw=True):
    if raw :
        imsave(path + "%s_original.png" % (ind), original)
        imsave(path + "%s_groudtruth.png" % (ind), forged)
    imsave(path + "%s_prediction.png" % (ind), mask, cmap="gray")

def demo(self, model, img_path="demo/8_original.png", save_path="demo/"): # just a fake method for testing
        original=read_rgb_image(img_path)
        _,mask,ptime=decode_an_image_file(original,model)
        imsave(save_path+"mask.png",mask,cmap="gray")
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
