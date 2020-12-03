import os
import cv2
import numpy as np
from datasets.dataset_interface import DataSetInterface
from utils.data_utils import  getData
class CoverageLoader(DataSetInterface):
    def __init__(self):
        self.name = "COVERAGE"
        self.testset = "datasets/cover_test_single.txt"
        self.base_path = "../VERAGE/"
        self.datalist = "../sequence/COVERAGE_image_new.list"
        self.paired_results, self.len, self.preprocess = prepare_coverage_dataset()
        self.paired_train, self.paired_tests = getData(testfile=self.testset, paired_results=self.paired_results, base_path=self.base_path)
    def load_data(self):
        return self.paired_results, self.len, self.preprocess
    def load_testset(self):
        return self.paired_train, len(self.paired_train), self.preprocess
    def load_trainset(self):
        return self.paired_tests, len(self.paired_tests), self.preprocess
    def get_name(self) -> str:
        return self.name
def prepare_coverage_dataset(input_image_file_list = "../sequence/COVERAGE_image_new.list"):
    with open(input_image_file_list, 'r') as IN:
        input_files = [line.strip() for line in IN.readlines()]
    print("INFO: successfully load", len(input_files), "input files")
    def get_input_ID(input_file):
        bname = os.path.basename(input_file)
        return bname.rsplit('.')[0]

    def get_mask_file_from_ID(sample_id):
        return os.path.join('../VERAGE/mask/', '{}forged.tif'.format(sample_id[:-1]))

    def preprocess(input_image, input_mask):
        x = input_image
        y = input_mask
        return x, y

    raw_lut = dict(zip([get_input_ID(f) for f in input_files], input_files))

    paired_results = []
    for key in raw_lut.keys():
        raw_file = raw_lut[key]
        mask_file = get_mask_file_from_ID(key)
        raw_file = '../' + raw_file

        r = cv2.imread(raw_file, 1)[..., ::-1]
        m = cv2.imread(mask_file, 0)
        m = np.zeros(m.shape)
        if r.shape[:2] != m.shape[:2]:
            continue
        raw_mask_dec = (raw_file, mask_file)
        paired_results.append(raw_mask_dec)

    print(len(paired_results))
    return paired_results, len(paired_results), preprocess
