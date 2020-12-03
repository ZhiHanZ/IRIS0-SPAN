import os

import cv2
from datasets.dataset_interface import DataSetInterface
class ColumbiaDataLoader(DataSetInterface):
    def __init__(self):
        self.name = "COLUMBIA"
        self.base_path = "../NIST16/"
        self.datalist = '../sequence/Columbia_image_new.list'
        self.paired_results, self.len, self.preprocess = prepare_columbia_dataset(input_image_file_list=self.datalist)
    def load_data(self):
        return self.paired_results, self.len, self.preprocess
    def load_trainset(self):
        print("no data for training in columbia")
        pass
    def load_testset(self):
        return self.paired_results, self.len, self.preprocess
    def get_name(self) -> str:
        return self.name
def prepare_columbia_dataset(input_image_file_list = '../sequence/Columbia_image_new.list', ):
    with open(input_image_file_list, 'r') as IN:
        input_files = [line.strip() for line in IN.readlines()]
    print("INFO: successfully load", len(input_files), "input files")

    def get_input_ID(input_file):
        bname = os.path.basename(input_file)
        return bname.rsplit('.')[0]

    def get_mask_file_from_ID(sample_id):
        return os.path.join('../Columbia/mixed/edgemask/', '{}_edgemask.jpg'.format(sample_id))

    def preprocess(input_image, input_mask):
        h, w = input_image.shape[:2]
        r = 512. / min(h, w)
        # nh, nw = 224, 224
        nh, nw = int(h * r), int(w * r)
        x = cv2.resize(input_image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        y = cv2.resize(input_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        return x, y

    raw_lut = dict(zip([get_input_ID(f) for f in input_files], input_files))

    paired_results = []
    for key in raw_lut.keys():
        raw_file = raw_lut[key]
        mask_file = get_mask_file_from_ID(key)

        raw_file = "../" + raw_file

        r = cv2.imread(raw_file, 1)[..., ::-1]
        m = cv2.imread(mask_file, 0)
        if r.shape[:2] != m.shape[:2]:
            continue
        raw_mask_dec = (raw_file, mask_file)
        paired_results.append(raw_mask_dec)

    return paired_results, len(paired_results), preprocess

