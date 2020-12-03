import os
import tensorflow as tf
import numpy as np
import cv2
import random
from datasets.dataset_interface import DataSetInterface

class CASIADataLoader(DataSetInterface):
    def __init__(self):
        self.name = "CASIA"
        self.test_root_path = "../CASIA1/"
        self.train_root_path = "../CASIA2/"
    def load_data(self):
        train, val, _, _ = prepare_casia_train(seed = 77, root_path = self.train_root_path)
        test, _, preprocess = self.load_testset()
        paired_res = train + val + test
        return paired_res, len(paired_res), preprocess
    def load_trainset(self):
        return prepare_casia_train(seed = 77, root_path = self.train_root_path)
    def load_testset(self):
        return prepare_casia_testset(root_path=self.test_root_path)
    def get_name(self) -> str:
        return self.name
def prepare_casia_testset(check=False, root_path = "../CASIA1/"):
    def preprocess(input_image, input_mask):
        x = input_image
        y = input_mask
        return x, y

    img_path = root_path + "Tp/"
    img_cm_path = img_path + "CM/"
    img_sp_path = img_path + "Sp/"
    mask_path = root_path + "mask/"
    mask_cm_path = mask_path + "CM/"
    mask_sp_path = mask_path + "Sp/"
    input_cm_file_list = "../CASIA1/Tp/Modified_CopyMove_list.txt"
    with open(input_cm_file_list, 'r') as IN:
        input_cm_files = [line.strip() for line in IN.readlines()]
        input_cm_dict = {path.split(".")[0]: img_cm_path + path for path in input_cm_files if
                         os.path.isfile(img_cm_path + path) == True}
    print("INFO: successfully load", len(input_cm_files), "input cm files")
    mask_cm_file_list = "../CASIA1/mask/CopyMove_groundtruth_list.txt"
    with open(mask_cm_file_list, 'r') as IN:
        mask_cm_files = [line.strip() for line in IN.readlines()]
        mask_cm_dict = {path.split(".")[0][:-3]: mask_cm_path + path for path in mask_cm_files if
                        os.path.isfile(mask_cm_path + path) == True}
    print("INFO: successfully load", len(mask_cm_files), "mask cm files")
    input_sp_file_list = "../CASIA1/Tp/Modified_Splicing_list.txt"
    with open(input_sp_file_list, 'r') as IN:
        input_sp_files = [line.strip() for line in IN.readlines()]
        input_sp_dict = {path.split(".")[0]: img_sp_path + path for path in input_sp_files if
                         os.path.isfile(img_sp_path + path) == True}
    print("INFO: successfully load", len(input_sp_files), "input sp files")
    mask_sp_file_list = "../CASIA1/mask/Splicing_groundtruth_list.txt"
    with open(mask_sp_file_list, 'r') as IN:
        mask_sp_files = [line.strip() for line in IN.readlines()]
        mask_sp_dict = {path.split(".")[0][:-3]: mask_sp_path + path for path in mask_sp_files if
                        os.path.isfile(mask_sp_path + path) == True}
        mask_sp_dict["Sp_D_NNN_A_ani0099_ani0100"] = mask_sp_path + "Sp_D_NNN_A_ani0099_ani0100.0288_gt.png"
    print("INFO: successfully load", len(mask_sp_files), "mask sp files")
    input_dict = {**input_cm_dict, **input_sp_dict}
    mask_dict = {**mask_cm_dict, **mask_sp_dict}
    paired_results = []
    for key in input_dict.keys():
        if key in mask_dict:
            if (check):
                raw_file = input_dict[key]
                mask_file = mask_dict[key]
                r = cv2.imread(raw_file, 1)[..., ::-1]
                m = cv2.imread(mask_file, 0)
                if r.shape[:2] != m.shape[:2]:
                    continue
            raw_mask_desc = (input_dict[key], mask_dict[key])
            paired_results.append(raw_mask_desc)
    print("length of paired_results is: ", len(paired_results))
    return paired_results, len(paired_results), preprocess

def prepare_casia_train(seed = 77, root_path = "../CASIA2/"):
    def preprocess( input_image, input_mask ) :
        x = np.expand_dims( input_image, axis=0 ).astype('float32')/255. * 2 - 1
        y = np.expand_dims( np.expand_dims( input_mask, axis=0 ), axis=-1 )/255.
        return x, y

    img_path = root_path + "Tp/"
    mask_path = root_path + "mask/"
    input_file_list = "../CASIA2/Tp/img_list.txt"
    with open(input_file_list, 'r') as IN:
        input_files = [line.strip() for line in IN.readlines()]
        input_dict = {path.split(".")[0] : img_path + path for path in input_files if os.path.isfile(img_path + path) == True}
    print ("INFO: successfully load", len( input_dict ), "input files")
    mask_list = "../CASIA2/mask/groundtruth_list.txt"
    with open(mask_list, 'r') as IN:
        mask_files = [line.strip() for line in IN.readlines()]
        mask_dict = {path.split(".")[0][:-3] : mask_path + path for path in mask_files if os.path.isfile(mask_path + path) == True}
    print ("INFO: successfully load", len( mask_dict ), "mask files")
    paired_results = []
    for key in input_dict.keys():
        if key in mask_dict:
            raw_file = input_dict[key]
            mask_file = mask_dict[key]
            r = cv2.imread( raw_file, 1 )[...,::-1]
            m = cv2.imread( mask_file, 0)
            if r.shape[:2] != m.shape[:2] :
                continue
            raw_mask_desc = (input_dict[key], mask_dict[key])
            paired_results.append(raw_mask_desc)
    print(len(paired_results))
    random.Random(seed).shuffle(paired_results)
    paired_cm_results = [elem for elem in paired_results if os.path.basename(elem[0])[3:4] == 'D']
    paired_sp_results = [elem for elem in paired_results if os.path.basename(elem[0])[3:4] == 'S']
    validation = paired_cm_results[:150] + paired_sp_results[:150]
    train = paired_cm_results[150:] + paired_sp_results[150:]
    print("INFO: Using %d images for train", len(train))
    print("INFO: Using %d images for train", len(validation))
    return train, validation, len(paired_cm_results + paired_sp_results), preprocess