import cv2
import numpy as np
from datetime import datetime


def read_rgb_image(image_file):
    rgb = cv2.imread(image_file, 1)[..., ::-1]
    return rgb


def decode_an_image_array(rgb, manTraNet):
    x = np.expand_dims(rgb.astype('float32') / 255. * 2 - 1, axis=0)
    t0 = datetime.now()
    y = manTraNet.predict(x)[0, ..., 0]
    t1 = datetime.now()
    return y, t1 - t0


def decode_an_image_file(rgb, manTraNet):
    mask, ptime = decode_an_image_array(rgb, manTraNet)
    return rgb, mask, ptime.total_seconds()


def postprocess(input_mask, shape):
    nh, nw = shape
    # r = 512./min(nh,nw)
    # nh, nw = int(nh*r), int(nw*r)
    mask = cv2.resize(input_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    # mask = np.expand_dims( mask, axis=0 )
    return mask
def RGB_normalization(img, reverse = False):
    if reverse:
        img = 1 - img/255.
    else :
        img = img/255.
    return img