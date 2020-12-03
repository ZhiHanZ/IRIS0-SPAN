from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import  numpy as np
import tensorflow as tf


def np_F1(y_true, y_pred):
    score = []
    for yy_true, yy_pred in zip(y_true, y_pred):
        this = f1_score((yy_true > .5).astype('int').ravel(), (yy_pred > .5).astype('int').ravel())
        that = f1_score((yy_true > .5).astype('int').ravel(), (1 - yy_pred > .5).astype('int').ravel())
        score.append(max(this, that))
    return np.mean(score).astype('float32')


def F1(y_true, y_pred):
    return tf.py_func(np_F1, [y_true, y_pred], 'float32')


def np_auc(y_true, y_pred):
    score = []
    for yy_true, yy_pred in zip( y_true, y_pred ) :
        try:
            this = roc_auc_score( (yy_true>.5).astype('int').ravel(), yy_pred.ravel() )
            that = roc_auc_score( (yy_true>.5).astype('int').ravel(), 1-yy_pred.ravel() )
            score.append( max( this, that ) )
        except:
            pass
    return np.mean( score ).astype('float32')


def np_F1_k(y_true, y_pred, k):
    score = []
    for yy_true, yy_pred in zip(y_true, y_pred):
        this = f1_score((yy_true > 0.5).astype('int').ravel(), (yy_pred > k).astype('int').ravel())
        that = f1_score((yy_true > 0.5).astype('int').ravel(), (1 - yy_pred > k).astype('int').ravel())
        score.append(max(this, that))
    return np.mean(score).astype('float32')


def np_ap(y_true, y_pred):
    score = []
    for yy_true, yy_pred in zip(y_true, y_pred):
        this = average_precision_score((yy_true > .5).astype('int').ravel(), yy_pred.ravel())
        that = average_precision_score((yy_true > .5).astype('int').ravel(), 1 - yy_pred.ravel())
        score.append(max(this, that))
    return np.mean(score).astype("float32")


def F1_k(y_true, y_pred):
    return tf.py_func(np_F1_k, [y_true, y_pred], 'float32')


def auroc(y_true, y_pred):
    return tf.py_func(np_auc, [y_true, y_pred], 'float32')

def compute_all(y_true, y_pred, k = 0.1):
    f1 = np_F1(y_true, y_pred)
    roc = np_auc(y_true, y_pred)
    f1_k = np_F1_k(y_true, y_pred, 0.1)
    ap = np_ap(y_true, y_pred)
    return {"f1" : f1, "AUC": roc, "f1_%d"%k : f1_k, "ap" : ap}
class DatasetF1Metrics :
    def __init__( self, dataset_list, engine_bsize_list ) :
        idx1_list = np.cumsum( engine_bsize_list )
        idx0_list = [0] + idx1_list[:-1].tolist()
        self.metrics = self._initialize( dataset_list, idx0_list, idx1_list )
    def _initialize( self, dataset_list, idx0_list, idx1_list ) :
        metrics = []
        for name, idx0, idx1 in zip( dataset_list, idx0_list, idx1_list ) :
            sname = name.split('_')[1]
            sdim = '256' if '256' in sname else '1024'
            new_name = "{}{}{}F1".format(name[:2], sname[:2].upper(), sdim )
            func_def = \
                "def {}( y_true, y_pred ) :\n    return F1( y_true[{}:{}], y_pred[{}:{}] )".format( new_name, idx0, idx1, idx0, idx1 )
            exec( func_def )
            print (func_def)
            metrics.append( locals()[new_name] )
        return metrics
