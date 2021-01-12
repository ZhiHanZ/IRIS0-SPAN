import sys
sys.path.insert(0,'/nas/home/yue_wu/ipynb/singleSplice/SignatureNet/network')
from layers.signatureNet import Conv2DSymPadding
from keras.layers import *
from keras.models import *
from keras.optimizers import *


class FeatureDeviation( Layer ) :
    """Samplewise Feature Euclidean Deviation Layer                                                                                                   
                                                                                                                                                      
    # Arguments:                                                                                                                                      
        scaling = bool, whether or not apply scaling in normalization                                                                                 
                                                                                                                                                      
                                                                                                                                                      
    # Note:                                                                                                                                           
        This layer normalizes an input feature tensor (4D) w.r.t. to sample's                                                                         
        mean and standard deviation. When abnormality exists, the corresponding                                                                       
        features will deviate much more from the mean feature than those not.                                                                         
    """
    def __init__( self, scaling=True, mode='thresh', use_abs=True, use_square=True, min_std=1e-5, cutoff=10, **kwargs ) :
        self.scaling = scaling
        self.mode = mode
        self.min_std_val = min_std
        self.cutoff = cutoff
        self.use_abs = use_abs
        self.use_square = use_square
        print ("INFO: mode =", mode, "min_std =", min_std, "cutoff =", cutoff)
        super( FeatureDeviation, self ).__init__( **kwargs )
    def build( self, input_shape ) :
        nb_feats = input_shape[-1]
        std_shape = ( 1,1,1, nb_feats )
        if self.scaling :
            self.min_std = self.add_weight( shape=std_shape,
                                            initializer=initializers.Constant(self.min_std_val),
                                            name='min_std',
                                            constraint=constraints.non_neg() )
            if ( self.mode == 'thresh' ) :
                self.max_dev = self.add_weight( shape=std_shape,
                                                initializer=initializers.Constant( self.cutoff ), # 10                                                
                                                name='max_dev',
                                                constraint=constraints.non_neg() )
                print( "INFO: mode=thresh, create trainable params: max_dev, min_std")
            elif ( self.mode == 'tanh') :
                self.alpha = self.add_weight( shape=std_shape,
                                              initializer=initializers.Constant( self.cutoff ), # 1.                                                  
                                              name='scaling_coef',
                                              constraint=constraints.non_neg() )
                print("INFO: mode=tanh, create trainable params: alpha, min_std")
        self.built = True
        return
    def call( self, x ) :
        mu = K.mean( x, axis=(1,2), keepdims=True )
        if self.scaling :
            sigma = K.maximum( K.std( x, axis=(1,2), keepdims=True ), K.epsilon() + self.min_std )
            xn = ( x-mu ) / sigma
            if ( self.mode == 'thresh' ) :
                diff = K.minimum( K.maximum( xn, -self.max_dev ), self.max_dev )
            elif ( self.mode == 'tanh' ) :
                diff = K.tanh( xn * self.alpha )
            else :
                print( "WARNING: unknown working mode {}".format( mode ) )
        else :
            diff = x - mu
        # apply abs if necessary                                                                                                                      
        if self.use_abs :
            diff = K.abs( diff )
        if self.use_square :
            diff = K.square( diff )
        return diff
    def compute_output_shape( self, input_shape ) :
        return input_shape


def bn_conv2d( x, nfilters, kernal_size, name, use_bn=False, use_relu=False, kernel_regularizer=None, use_bias=True ) :
    '''basic Conv2D+BatchNorm+ReLu module                                                                                                             
    '''
    x = Conv2DSymPadding( nfilters,
                          kernal_size,
                          padding='same',
                          activation=None,
                          use_bias=use_bias,
                          kernel_regularizer=kernel_regularizer,
                          name=name+'-conv' )(x)
    if ( use_bn ) :
        x = BatchNormalization( name=name+'-bnorm', momentum=.995 )(x)
    if ( use_relu ) :
        x = Activation('relu', name=name+'-relu' )(x)
    return x


def compute_one_class_average_feature( f, m ) :
    mu = K.sum( f * m, axis=(1,2), keepdims=True ) / ( K.sum( m, axis=(1,2), keepdims=True ) + K.epsilon() )
    #mu = K.l2_normalize( mu, axis=-1 )
    return mu

def betweenLoss( y_true, y_pred ) :
    mask = y_true
    feat = y_pred
    # 1. compute foreground class average feature                                                                                                     
    frg_mu = compute_one_class_average_feature( feat, mask )
    bkg_mu = compute_one_class_average_feature( feat, 1-mask )
    # 2. compute between class loss                                                                                                                   
    frg_loss = bkg_mu * feat * mask
    bkg_loss = frg_mu * feat * (1-mask)
    # 3. overall loss                                                                                                                                 
    loss = frg_loss + bkg_loss
    return K.sum( loss, axis=-1 )

def withinLoss( y_true, y_pred ) :
    mask = y_true
    feat = y_pred
    # 1. compute foreground class average feature                                                                                                     
    frg_mu = compute_one_class_average_feature( feat, mask )
    bkg_mu = compute_one_class_average_feature( feat, 1-mask )
    # 2. compute between class loss                                                                                                                   
    frg_loss = K.square(frg_mu - feat) * mask
    bkg_loss = K.square(bkg_mu - feat) * (1-mask)
    # 3. overall loss                                                                                                                                 
    loss = frg_loss + bkg_loss
    return K.sum( loss, axis=-1 )


def batch_triplet( y_true, y_pred ) :
    mask = y_true
    feat = y_pred
    # 1. compute foreground class average feature                             
    frg_mu = compute_one_class_average_feature( feat, mask )
    bkg_mu = compute_one_class_average_feature( feat, 1-mask )
    # 2. compute between class loss
    pos_var = K.square(frg_mu - feat) * mask + K.square(bkg_mu - feat) * (1-mask)
    neg_var = K.square(frg_mu - feat) * (1-mask) + K.square(bkg_mu - feat) * mask
    # 3. overall loss                                                                                                           
    loss = K.maximum( pos_var-neg_var + 0.5, 0 )
    #loss = (K.square(frg_mu - feat) - K.square(bkg_mu - feat)) * (2*mask-1)
    #loss = K.maximum( loss+0.5, 0)
    return K.mean( loss, axis=-1 )

def var( y_true, y_pred ) :
    return K.sum( K.var( y_pred, axis=(1,2), keepdims=True ), axis=-1 )

def create_dnlm_model( nb_mani_classes=9,
                  nb_camera_classes=15,
                  input_shape=(None,None,3),
                  name='SignatureNet',
                  kernel_size=(3,3),
                  nb_filters=64,
                  nb_outputs=1,
                  nb_layers=17,
                  use_bn=True,
                  use_relu=True,
                  decision_size=(7,7)) :
    '''SignatureNet
    '''
    obs = Input( shape=input_shape, name=name+'-input' )
    x = Conv2DSymPadding( nb_filters, kernel_size,
                activation='relu',
                name=name+'-c1',
                padding='same' )( obs )
    for k in range(nb_layers-3) :
        x = bn_conv2d( x, nb_filters, kernel_size,
                       name = name +'-c%d' % (k+2),
                       use_bn=use_bn,
                       use_relu=use_relu )
    # compute local statistics
    sf = Conv2DSymPadding( nb_filters*2, kernel_size,
                          activation=None,
                          name=name+'-transform',
                          padding='same' )(x)  
    sf = Lambda( lambda t : K.l2_normalize( t, axis=-1), name='FV')(sf)
    mc = Conv2DSymPadding( nb_mani_classes, decision_size, 
                          activation = 'softmax', 
                          name = 'MC', 
                          padding = 'same' )(sf)
    cc = Conv2DSymPadding( nb_camera_classes, decision_size, 
                          activation = 'softmax', 
                          name = 'CC', 
                          padding = 'same' )(sf)
    d = FeatureDeviation( mode='tanh',
                          use_abs=True,
                          min_std=1/32.,
                          use_square=False,
                          cutoff=2,
                          name=name+'-dev' )(sf)
    mm = Conv2DSymPadding( nb_outputs, decision_size,
                           activation='sigmoid',
                           name='MM',
                           padding='same' )(d)
    return Model( inputs=obs, output=[mm, mc, cc, sf], name=name )

def dnlm17P3( label_mode=1 ) :
    # 16 * 3 - 1 = 47
    nb_mani_classes = 9 if label_mode==1 else 55
    return create_dnlm_model( nb_mani_classes=9,
                  nb_camera_classes=15,
                  input_shape=(None,None,3),
                  name='SignatureNet',
                  kernel_size=(3,3),
                  nb_filters=64,
                  nb_outputs=1,
                  nb_layers=17,
                  use_bn=True,
                  use_relu=True,
                  decision_size=(3,3))

def dnlm13P5( label_mode=1 ) :
    # 12 * 5 - 1 = 59
    nb_mani_classes = 9 if label_mode==1 else 55
    return create_dnlm_model( nb_mani_classes=9,
                  nb_camera_classes=15,
                  input_shape=(None,None,3),
                  name='SignatureNet',
                  kernel_size=(5,5),
                  nb_filters=64,
                  nb_outputs=1,
                  nb_layers=13,
                  use_bn=True,
                  use_relu=True,
                  decision_size=(5,5))


def vgg16( label_mode=1, base=32 )  :
    nb_mani_classes = 9 if label_mode==1 else 55
    nb_camera_classes = 15
    decision_size = (3,3)
    img_input = Input(shape=(None,None,3), name='image_in')
    # block 1
    bname = 'b1' # 32
    nb_filters = base
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( img_input )
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    # block 2
    bname = 'b2'
    nb_filters = 2 * base # 64
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( x )
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    # block 3
    bname = 'b3'
    nb_filters = 3 * base # 96
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( x )
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c3')( x )
    # block 4
    bname = 'b4'
    nb_filters = 4 * base # 128
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( x )
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    #x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c3')( x )
    # block 5/bottle-neck
    sf = Conv2DSymPadding( nb_filters, (3,3),
                          activation=None,
                          name='transform',
                          padding='same' )(x)  
    sf = Lambda( lambda t : K.l2_normalize( t, axis=-1), name='FV')(sf)
    mc = Conv2DSymPadding( nb_mani_classes, decision_size, 
                          activation = 'softmax', 
                          name = 'MC', 
                          padding = 'same' )(sf)
    cc = Conv2DSymPadding( nb_camera_classes, decision_size, 
                          activation = 'softmax', 
                          name = 'CC', 
                          padding = 'same' )(sf)
    d = FeatureDeviation( mode='tanh',
                          use_abs=True,
                          min_std=1/32.,
                          use_square=False,
                          cutoff=2,
                          name='featDev' )(sf)
    mm = Conv2DSymPadding( 1, decision_size,
                           activation='sigmoid',
                           name='MM',
                           padding='same' )(d)
    return Model( inputs=img_input, output=[mm, mc, cc, sf], name='vgg16' )


def residual_block( xin, nfilters, kernal_size, name, kernel_regularizer=None ) :
    x = Conv2DSymPadding( nfilters,
                          kernal_size,
                          padding='same',
                          activation=None,
                          kernel_regularizer=kernel_regularizer,
                          name=name+'-conv1' )(xin)
    x = BatchNormalization( name=name+'-bnorm1' )(x)
    x = Activation('relu', name=name+'-relu1' )(x)
    x = Conv2DSymPadding( nfilters,
                      kernal_size,
                      padding='same',
                      activation=None,
                      kernel_regularizer=kernel_regularizer,
                      name=name+'-conv2' )(x)
    x = BatchNormalization( name=name+'-bnorm2' )(x)
    if K.int_shape(x)[-1] != K.int_shape(xin)[-1] :
        y = Conv2DSymPadding( nfilters,
                              (1,1),
                              padding='same',
                              activation=None,
                              kernel_regularizer=kernel_regularizer,
                              name=name+'-convt' )(xin)
    else :
        y = xin
    x = Add(name=name+'-merge')([x, y])
    x = Activation('relu', name=name+'-relu2' )(x)
    return x

def create_resnet_model( nb_res_filters=64, nb_mani_classes=9, input_shape=(256,256,3)) :
    nb_camera_classes = 15
    def create_resnet18(input_shape) :
        net_in = Input(shape=input_shape, name='img_in')
        # block 1
        x = bn_conv2d( net_in, 1*nb_res_filters, (7,7), name='blk1', use_bias=False, use_relu=True, use_bn=True )
        # block 2
        x = residual_block( x, 1*nb_res_filters, (3,3), name='blk2' )
        # block 3
        x = residual_block( x, 2*nb_res_filters, (3,3), name='blk3' )
        # block 4
        x = residual_block( x, 4*nb_res_filters, (3,3), name='blk4' )
        # block 5
        net_out = residual_block( x, 8*nb_res_filters, (3,3), name='blk5' )
        return Model( inputs=net_in, outputs=net_out, name='featex' )
    # 1. create featex
    ResNet18 = create_resnet18(input_shape)
    # 2. create network
    img_in    = Input(shape=input_shape, name='img_in')
    # 2.b feature extraction
    x = ResNet18( img_in )
    sf = Conv2D( 8*nb_res_filters, (3,3), activation=None, name='trans', padding='same', use_bias=False )(x)
    sf = Lambda( lambda t : K.l2_normalize( t, axis=-1), name='FV')(sf)
    decision_size=(3,3)
    mc = Conv2DSymPadding( nb_mani_classes, decision_size, 
                          activation = 'softmax', 
                          name = 'MC', 
                          padding = 'same' )(sf)
    cc = Conv2DSymPadding( nb_camera_classes, decision_size, 
                          activation = 'softmax', 
                          name = 'CC', 
                          padding = 'same' )(sf)
    d = FeatureDeviation( mode='tanh',
                          use_abs=True,
                          min_std=1/32.,
                          use_square=False,
                          cutoff=2,
                          name='featDev' )(sf)
    mm = Conv2DSymPadding( 1, decision_size,
                           activation='sigmoid',
                           name='MM',
                           padding='same' )(d)
    return Model( inputs=img_in, output=[mm, mc, cc, sf], name='resnet18' )

def resnet18( label_mode=1, nb_filters=18 ) :
    nb_mani_classes = 9 if label_mode==1 else 55
    return create_resnet_model( nb_filters, nb_mani_classes )
    
import keras
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras import backend as K

def contLoss( y_true, y_pred ) :
    loss = betweenLoss( y_true, y_pred )
    loss += withinLoss( y_true, y_pred )
    return loss

def vgg16AC( label_mode=1, base=32 )  :
    nb_mani_classes = 9 if label_mode==1 else 55
    nb_camera_classes = 15
    decision_size = (3,3)
    img_input = Input(shape=(None,None,3), name='image_in')
    # block 1
    bname = 'b1' # 32
    nb_filters = base
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( img_input )
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    # block 2
    bname = 'b2'
    nb_filters = 2 * base # 64
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( x )
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    # block 3
    bname = 'b3'
    nb_filters = 3 * base # 96
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( x )
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c3')( x )
    # block 4
    bname = 'b4'
    nb_filters = 4 * base # 128
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( x )
    x = Conv2D( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    # block 5/bottle-neck
    sf = Conv2DSymPadding( nb_filters, (3,3),
                          activation=None,
                          name='transform',
                          padding='same' )(x)  
    sf = Lambda( lambda t : K.l2_normalize( t, axis=-1), name='FV')(sf)
    ac = Conv2DSymPadding( 22, decision_size, 
                          activation = 'hard_sigmoid', 
                          name = 'AC', 
                          padding = 'same' )(sf)
    d = FeatureDeviation( mode='tanh',
                          use_abs=True,
                          min_std=1/32.,
                          use_square=False,
                          cutoff=2,
                          name='featDev' )(sf)
    mm = Conv2DSymPadding( 1, decision_size,
                           activation='hard_sigmoid',
                           name='MM',
                           padding='same' )(d)
    return Model( inputs=img_input, output=[mm, ac, sf], name='vgg16' )

class SGDAccumulate(keras.optimizers.Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, accum_iters=1, **kwargs):
        super(SGDAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.accum_iters = K.variable(accum_iters)
        self.initial_decay = decay
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        accum_switch = K.equal(self.iterations % self.accum_iters, 0)
        accum_switch = K.cast(accum_switch, dtype='float32')

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        temp_grads = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, cg, m, tg in zip(params, grads, moments, temp_grads):
            g = cg + tg
            v = self.momentum * m - (lr * g / self.accum_iters)  # velocity
            self.updates.append(K.update(m, (1 - accum_switch) * m + accum_switch * v))
            self.updates.append(K.update(tg, (1 - accum_switch) * g))

            if self.nesterov:
                new_p = p + self.momentum * v - (lr * g / self.accum_iters)
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - accum_switch) * p + accum_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'accum_iters': int(K.get_value(self.accum_iters)) }
        base_config = super(SGDAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class AdamAccumulate(Optimizer):   
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, accum_iters=2, **kwargs):
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, dtype='int64')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, gg in zip(params, grads, ms, vs, vhats, gs):

            flag = K.equal(self.iterations % self.accum_iters, 0)
            flag = K.cast(flag, K.floatx())

            gg_t = (1 - flag) * (gg + g)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (gg + flag * g) / K.cast(self.accum_iters, K.floatx())
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square((gg + flag * g) / K.cast(self.accum_iters, K.floatx()))
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append((m, flag * m_t + (1 - flag) * m))
            self.updates.append((v, flag * v_t + (1 - flag) * v))
            self.updates.append((gg, gg_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))