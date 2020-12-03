import cv2
import numpy as np
from datasets.dataset_interface import DataSetInterface
from utils.data_utils import  getData
class NISTLoader(DataSetInterface):
    def __init__(self):
        self.name = "NIST"
        self.testset = "datasets/NIST_test_single.txt"
        self.base_path = "../NIST16/"
        self.datalist = '../NIST16/mani_new.list'
        self.paired_results, self.len, self.preprocess = prepare_nist_dataset()
        self.paired_tests, self.paired_train = getData(testfile=self.testset, paired_results=self.paired_results, base_path=self.base_path)
    def load_data(self):
        return self.paired_results, self.len, self.preprocess
    def load_trainset(self):
        return self.paired_train, len(self.paired_train), self.preprocess
    def load_testset(self):
        return self.paired_tests, len(self.paired_tests), self.preprocess
    def get_name(self) -> str:
        return self.name
def prepare_nist_dataset( check=False , input_image_file_list = '../NIST16/mani_new.list', base_path="../NIST16/") :
    with open( input_image_file_list, 'r') as IN :
        input_files = [ line.strip().split( ) for line in IN.readlines() ]
    print ("INFO: successfully load", len( input_files ), "input files")
    def preprocess( input_image, input_mask ) :
        h, w = input_image.shape[:2]
        r = 512./min(h,w)
        nh, nw = int(h*r), int(w*r)
        x = cv2.resize(input_image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        #y =  cv2.resize(1-input_mask/255, (nw,nh), interpolation=cv2.INTER_NEAREST)
        y = cv2.resize(1 - input_mask/255., (nw, nh), interpolation=cv2.INTER_NEAREST)
        return x, y
    paired_results = []
    for raw_file, mask_file in input_files :
        if check :
            r = cv2.imread(base_path + raw_file, 1 )[...,::-1]
            m = cv2.imread(base_path + mask_file, 0)
            if r.shape[:2] != m.shape[:2] :
                continue
        raw_mask_dec = (base_path + raw_file,base_path + mask_file )
        paired_results.append( raw_mask_dec )
    return paired_results, len(paired_results), preprocess
