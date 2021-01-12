from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import parse

print ("INFO: use utils version=3")

from keras.callbacks import Callback
from datetime import datetime, timedelta

class Timer( Callback ) :
    def __init__( self, time_limit = '23:59:00' ) :
        self.time_start = datetime.now()
        self.elapse_sofar = None
        hours, minutes, seconds = [ int(v) for v in time_limit.split(':') ]
        self.time_limit = timedelta( hours = hours, minutes = minutes, seconds = seconds)
        self.time_of_epoch = [datetime.now()]
    def on_epoch_begin( self, epoch, logs = None  ) :
        now = datetime.now()
        if ( len( self.time_of_epoch ) == 0 ) :
            previous = self.time_start
        else :
            previous = self.time_of_epoch[-1]
        self.time_of_epoch.append( now )
        time_elapse = now - previous
        if (self.elapse_sofar is None):
            self.elapse_sofar = time_elapse
        else :
            self.elapse_sofar += time_elapse
        if ( time_elapse + self.elapse_sofar > self.time_limit ) :
            print ("Timer catch over time")
            raise
        return

def load_cached_weights( weight_dir, weight_prefix = None, mode = 'best', start_lr=1e-4 ) :
    lut = dict()
    for f in os.listdir( weight_dir ) :
        if f.endswith('.h5') :
            if ( weight_prefix is None ) or ( f.startswith( weight_prefix ) ) :
                h5 = os.path.join( weight_dir, f )
                mtime = os.stat( h5 ).st_mtime
                try :
                    weight_format = '{}_E{04d}-{2.f}-{2.f}.h5'
                    #print "[[[[[[[[[[[HERE]]]]]]]]]]]"
                    fields = parse.parse( weight_format, f )
                    _, epoch, loss, lr = fields
                    
                    epoch = int(epoch)
                    #print epoch
                    loss = float(loss)
                    
                    if np.isnan( loss ) :
                        print ("INFO: skip nan weights at 1", f)
                        loss = 1e9
                    elif ('nan' in f) :
                        print ("INFO: skip nan weights at 2", f)
                        loss = 1e9                        
                    lr = float(lr)
                except Exception as e :
                    print (f, e)
                    loss = 1e9
                    epoch = 0
                    lr = start_lr
                lut[ h5 ] = [ mtime, loss, epoch, lr ]
    keys = lut.keys()
    
    if ( keys ) :
        vals = np.row_stack( [ lut[k] for k in keys ] )
        if ( mode == 'best' ) :
            idx = vals[:,1].argmin()
        elif ( mode == 'latest' ) :
            idx = vals[:,0].argmax()
        init_weight_file = keys[idx]
        init_lr = vals[idx,3]
        init_epoch = int( vals[:,2].max() )
        print ("USING ",init_weight_file)
        return init_weight_file, init_epoch, init_lr
    else :
        return None, 0, start_lr

def prepare_callbacks( expt_root, model_name, testing_callback = None, patience=10, time_limit=None ) :
    weight_file = os.path.join( expt_root, '%s.h5' %  model_name )
    logger_file = os.path.join( expt_root, '%s.csv' % model_name )
    csv_logger = CSVLogger( logger_file, append = True )
    #ckpt1 = ModelCheckpoint( weight_file, verbose = 1, save_best_only = True , monitor='val_auroc', mode='max')
    ckpt1 = ModelCheckpoint( weight_file, verbose = 1, save_best_only = True, monitor='val_loss', mode='min')
    prefix = ".".join( weight_file.split('.')[:-1] )
    #ckpt2 = ModelCheckpoint( prefix + '_E{epoch:04d}-{val_auroc:.5f}-{lr:.7f}.h5', verbose = 1, period = 1 )
    ckpt2 = ModelCheckpoint( prefix + '_E{epoch:04d}-{val_loss:.5f}-{lr:.7f}.h5', verbose = 1, period = 1 )
    count_down = Timer() if time_limit is None else Timer( time_limit )
    board = TensorBoard( log_dir=expt_root, 
                         histogram_freq=0,
                         write_graph=False,
                         write_grads=False,
                         batch_size=32, 
                         write_images=True,)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience, min_lr=1e-7, verbose = 1, mode='min')
    #earlystop = EarlyStopping( patience=5*patience, verbose=1, monitor='val_auroc', mode='max')
    earlystop = EarlyStopping( patience=3*patience, verbose=1)
    my_callbacks = [ earlystop, reduce_lr, ckpt1, ckpt2, csv_logger, board, count_down ]
    if ( testing_callback is not None ) :
        my_callbacks += [ testing_callback ]
    return my_callbacks

from keras.losses import binary_crossentropy
def rec( y_true, y_pred ):
    y_true_mask = K.cast( y_true > 0.5, 'float32' )
    return K.sum( K.cast((y_pred * y_true_mask) > 0.5, "float32"), axis=(1,2) ) / (K.sum( y_true_mask, axis=(1,2) ) + .1)

def pre( y_true, y_pred ):
    y_pred_mask = K.cast( y_pred > 0.5, 'float32' )
    return K.sum( K.cast((y_true * y_pred_mask) > 0.5, "float32"), axis=(1,2) ) / (K.sum( y_pred_mask, axis=(1,2) ) + .1)

def F1( y_true, y_pred ):
    recall_ = rec( y_true, y_pred )
    precision_ = pre( y_true, y_pred )
    return 2*recall_*precision_ / ( recall_ + precision_ + K.epsilon() )

def modified_binary_entropy( y_true, y_pred, coef = 4 ):
    loss = binary_crossentropy( y_true, y_pred )
    print (K.int_shape(loss))
    mask = K.cast( y_true > .5, 'float32')
    mask = K.squeeze(mask,axis=-1)
    print (K.int_shape(mask))
    return (1 + coef * mask) * loss


