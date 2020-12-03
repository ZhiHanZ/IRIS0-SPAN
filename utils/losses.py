import keras.backend as K
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.keras.losses import Loss
# focal did not perform well in our training
def _focal(y_true, y_pred, alpha =0.25, gamma = 2, axis = -1):
    epsilon_ = constant_op.constant(K.epsilon(), dtype=y_pred.dtype.base_dtype) #1e-7
    y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1 - epsilon_)
    fl = y_true * tf.math.pow(1 - y_pred, gamma)*alpha*math_ops.log(y_pred + K.epsilon())
    fl += (1 - y_true)*tf.math.pow(y_pred, gamma)*(1 - alpha)*math_ops.log(1 - y_pred + K.epsilon())
    return -fl

def focal(y_true, y_pred, alpha = 0.25, gamma = 2):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(_focal(y_true, y_pred, alpha, gamma), axis=-1)
class Focal(Loss):
    def __init__(self, fn=focal, reduction=losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE, alpha =0.25, gamma = 2, name="focal_loss" ):
        super(Focal, self).__init__(
            name=name,
            reduction=reduction,
        )
        self.alpha= alpha
        self.gamma = gamma
    def call(self, y_true, y_pred):
        return focal(y_true, y_pred, alpha=self.alpha, gamma= self.gamma)
# with tf.compat.v1.Session() as sess:
#     lossFunc = Focal(alpha=0.5, gamma=0)
#     loss = lossFunc([0., 0., 1., 1.], [1., 1., 1., 0.])
#     print(loss.eval())