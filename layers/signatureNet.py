from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.losses import *
import tensorflow as tf
from keras.layers.convolutional import _Conv
from keras.legacy import interfaces
from keras.engine import InputSpec
from keras import backend as K
from keras.constraints import Constraint

class Conv2DSymPadding( _Conv ) :
    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 padding='same',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv2DSymPadding, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)
    def get_config(self):
        config = super(Conv2DSymPadding, self).get_config()
        config.pop('rank')
        return config
    def call( self, inputs ) :
        if ( isinstance( self.kernel_size, tuple ) ) :
            kh, kw = self.kernel_size
        else :
            kh = kw = self.kernel_size
        ph, pw = kh//2, kw//2
        inputs_pad = tf.pad( inputs, [[0,0],[ph,ph],[pw,pw],[0,0]], mode='symmetric' )
        outputs = K.conv2d(
                inputs_pad,
                self.kernel,
                strides=self.strides,
                padding='valid',
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class SRMConv2D( Layer ) :
    """SRM filter layer
    """
    def _get_srm_list( self ) :
        # srm kernel 1
        srm1 = np.zeros([5,5]).astype('float32')
        srm1[1:-1,1:-1] = np.array([[-1, 2, -1],
                                    [2, -4, 2],
                                    [-1, 2, -1]] )
        srm1 /= 4.
        # srm kernel 2              
        srm2 = np.array([[-1, 2, -2, 2, -1],
                         [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2],
                         [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]]).astype('float32')
        srm2 /= 12.                                    
        # srm kernel 3
        srm3 = np.zeros([5,5]).astype('float32')
        srm3[2,1:-1] = np.array([1,-2,1])
        srm3 /= 2.
        return [ srm1, srm2, srm3 ]       
    def build( self, input_shape ) :
        bsize, nb_rows, nb_cols, nb_feats = input_shape
        assert nb_feats == 3, "ERROR: SRM only accepts a RGB/3-channel input"
        kernel = []
        srm_list = self._get_srm_list()
        for idx, srm in enumerate( srm_list ):
            for ch in range(3) :
                this_ch_kernel = np.zeros([5,5,3]).astype('float32')
                this_ch_kernel[:,:,ch] = srm
                kernel.append( this_ch_kernel )
        kernel = np.stack( kernel, axis=-1 )
        self.kernel = K.variable( kernel, dtype='float32', name='srm' )
        self.kernel_size = (5,5)
        self.built = True
        return
    def call( self, inputs ) :
        if ( isinstance( self.kernel_size, tuple ) ) :
            kh, kw = self.kernel_size
        else :
            kh = kw = self.kernel_size
        ph, pw = kh//2, kw//2
        inputs_pad = tf.pad( inputs, [[0,0],[ph,ph],[pw,pw],[0,0]], mode='symmetric' )
        outputs = K.conv2d(
                inputs_pad,
                self.kernel,
                strides=(1,1),
                padding='valid',
                data_format='channels_last',
                dilation_rate=(1,1))
        return outputs
    def compute_output_shape( self, input_shape ) :
        return input_shape[:3] + (9,)

class BayarConstraint( Constraint ) :
    def __init__( self ) :
        self.mask = None
    def _initialize_mask( self, w ) :
        nb_rows, nb_cols, nb_inputs, nb_outputs = K.int_shape(w)
        m = np.zeros([nb_rows, nb_cols, nb_inputs, nb_outputs]).astype('float32')
        m[nb_rows//2,nb_cols//2] = 1.
        self.mask = K.variable( m, dtype='float32' )
        return
    def __call__( self, w ) :
        if self.mask is None :
            self._initialize_mask(w)
        w *= (1-self.mask)
        rest_sum = K.sum( w, axis=(0,1), keepdims=True)
        w /= rest_sum + K.epsilon()
        w -= self.mask
        return w
    
class BayarConv2D( Conv2DSymPadding ) :
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 padding='same',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv2DSymPadding, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=BayarConstraint(),
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)
        

class CombinedConv2D( Conv2DSymPadding ) :
    def __init__(self, filters,
                 kernel_size=(5,5),
                 strides=(1,1),
                 data_format=None,
                 dilation_rate=(1,1),
                 activation=None,
                 padding='same',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(CombinedConv2D, self).__init__(
            filters=filters,
            kernel_size=(5,5),
            strides=strides,
            padding='same',
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=None,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=None,
            **kwargs)        
        self.input_spec = InputSpec(ndim=4)
    def _get_srm_list( self ) :
        # srm kernel 1
        srm1 = np.zeros([5,5]).astype('float32')
        srm1[1:-1,1:-1] = np.array([[-1, 2, -1],
                                    [2, -4, 2],
                                    [-1, 2, -1]] )
        srm1 /= 4.
        # srm kernel 2              
        srm2 = np.array([[-1, 2, -2, 2, -1],
                         [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2],
                         [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]]).astype('float32')
        srm2 /= 12.                                    
        # srm kernel 3
        srm3 = np.zeros([5,5]).astype('float32')
        srm3[2,1:-1] = np.array([1,-2,1])
        srm3 /= 2.
        return [ srm1, srm2, srm3 ] 
    def _build_SRM_kernel( self ) :
        kernel = []
        srm_list = self._get_srm_list()
        for idx, srm in enumerate( srm_list ):
            for ch in range(3) :
                this_ch_kernel = np.zeros([5,5,3]).astype('float32')
                this_ch_kernel[:,:,ch] = srm
                kernel.append( this_ch_kernel )
        kernel = np.stack( kernel, axis=-1 )
        srm_kernel = K.variable( kernel, dtype='float32', name='srm' )
        return srm_kernel
    def build( self, input_shape ) :
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        # 1. regular conv kernels, fully trainable
        filters = self.filters - 9 - 3
        if filters >= 1 :
            regular_kernel_shape = self.kernel_size + (input_dim, filters)
            self.regular_kernel = self.add_weight(shape=regular_kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='regular_kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        else :
            self.regular_kernel = None
        # 2. SRM kernels, not trainable
        self.srm_kernel = self._build_SRM_kernel()
        # 3. bayar kernels, trainable but under constraint
        bayar_kernel_shape = self.kernel_size + (input_dim, 3)
        self.bayar_kernel = self.add_weight(shape=bayar_kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='bayar_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=BayarConstraint())
        # 4. collect all kernels
        if ( self.regular_kernel is not None ) :
            all_kernels = [ self.regular_kernel, 
                            self.srm_kernel,
                            self.bayar_kernel]
        else :
            all_kernels = [ self.srm_kernel,
                            self.bayar_kernel]
        self.kernel = K.concatenate( all_kernels, axis=-1 )
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

class ContentConv2D( _Conv ) :
    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 nb_inputs,
                 strides=(1,1),
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 padding='same',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.psize = np.product( kernel_size )
        self.fsize = filters
        super(ContentConv2D, self).__init__(
            rank=2,
            filters=filters*np.prod( kernel_size )*nb_inputs,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)
    def call( self, inputs ) :
        batch_size, nb_rows, nb_cols, nb_inputs = [ K.shape( inputs )[k] for k in range(4) ]
        # 1. predict parameters
        params = K.conv2d( inputs, 
                           self.kernel,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate )
        #print "params.shape =", K.int_shape( params ) # BxHxWx(Px#inputsx#outputs)
        params = K.reshape( params, [batch_size, nb_rows, nb_cols, -1, self.fsize ] )
        #print "params.shape =", K.int_shape( params ) # BxHxWx(Px#inputsx#outputs)
        # 2. prepare patches
        nb_feats = K.int_shape( inputs )[-1]
        if ( isinstance( self.kernel_size, tuple ) ) :
            kh, kw = self.kernel_size
        else :
            kh = kw = self.kernel_size
        ph, pw = kh//2, kw//2
        inputs_pad = tf.pad( inputs, [[0,0],[ph,ph],[pw,pw],[0,0]], mode='symmetric' )        
        patches = tf.extract_image_patches( inputs_pad,
                                            ksizes=[1, kh, kw, 1],
                                            strides=[1,1,1,1],
                                            rates=[1,1,1,1],
                                            padding='VALID' )
        #print "patches.shape =", K.int_shape( patches ) # BxHxWx(Px#inputs)
        patches = K.expand_dims( patches, axis=-1 )
        # 3. perform patchwise conv
        outputs = K.sum( params * patches, axis=-2, keepdims=False )
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    def compute_output_shape( self, input_shape ) :
        return input_shape[:3] + ( self.fsize, )
        
####################################################################################################
# SignatureNet Layers
####################################################################################################
class EuclideanDeviation( Layer ) :
    """Samplewise Euclidean Deviation Layer

    # Arguments:
        centering = bool, whether or not apply centering in normalization
        scaling = bool, whether or not apply scaling in normalization
        min_std = float, the minimum std used for each feature dimension
        max_dev = float, the maximum of deviation in consideration

    # Note:
        This layer normalizes an input feature tensor (4D) w.r.t. to sample's
        mean and standard deviation. When abnormality exists, the corresponding
        features will deviate much more from the mean feature than those not.
    """
    def __init__( self, centering=False, scaling=False, min_std=1e-5, max_dev=10, **kwargs ) :
        self.min_std = min_std
        self.max_dev = max_dev
        self.scaling = scaling
        super( EuclideanDeviation, self ).__init__( **kwargs )
    def call( self, x ) :
        mu = K.mean( x, axis=(1,2), keepdims=True )
        if self.scaling :
            sigma = K.maximum( K.std( x, axis=(1,2), keepdims=True ), self.min_std )
            xn = ( x-mu ) / sigma
            diff = K.abs( K.clip( xn, -self.max_dev, self.max_dev ) )
        else :
            diff = x - mu
        return diff
    def compute_output_shape( self, input_shape ) :
        return input_shape

class AngularDeviation( Layer ) :
    """Samplewise Angular Deviation Layer

    # Arguments:
        centering = bool, whether or not apply centering in normalization
        scaling = bool, whether or not apply scaling in normalization

    # Note:
        This layer normalizes an input feature tensor (4D) w.r.t. to sample's
        mean and standard deviation. When abnormality exists, the corresponding
        features will deviate much more from the mean feature than those not.
    """
    def __init__( self, centering=False, scaling=False, **kwargs ) :
        self.centering = centering
        super( AngularDeviation, self ).__init__( **kwargs )
    def call( self, x ) :
        mu = K.mean( x, axis=(1,2), keepdims=True )
        mu = K.l2_normalize( mu, axis=-1 )
        xn = K.l2_normalize( x, axis=-1 )
        diff = xn * mu
        if self.centering :
            mu_diff = K.mean( diff, axis=(1,2), keepdims=True )
            diff = diff - mu_diff
        return diff
    def compute_output_shape( self, input_shape ) :
        return input_shape

class FeatureDeviation( Layer ) :
    """Samplewise Feature Euclidean Deviation Layer

    # Arguments:
        scaling = bool, whether or not apply scaling in normalization


    # Note:
        This layer normalizes an input feature tensor (4D) w.r.t. to sample's
        mean and standard deviation. When abnormality exists, the corresponding
        features will deviate much more from the mean feature than those not.
    """
    def __init__( self, scaling=True, mode='thresh', use_abs=True, use_square=True, **kwargs ) :
        self.scaling = scaling
        self.mode = mode
        self.use_abs = use_abs
        self.use_square = use_square
        super( FeatureDeviation, self ).__init__( **kwargs )
    def build( self, input_shape ) :
        nb_feats = input_shape[-1]
        std_shape = ( 1,1,1, nb_feats )
        if self.scaling :
            self.min_std = self.add_weight( shape=std_shape,
                                            initializer=initializers.Constant(1e-5),
                                            name='min_std',
                                            constraint=constraints.non_neg() )
            if ( self.mode == 'thresh' ) :
                self.max_dev = self.add_weight( shape=std_shape,
                                                initializer=initializers.Constant(10),
                                                name='max_dev',
                                                constraint=constraints.non_neg() )
                self.min_dev = self.add_weight( shape=std_shape,
                                                initializer=initializers.Constant(10),
                                                name='min_dev',
                                                constraint=constraints.non_neg() )
        self.built = True
        return
    def call( self, x ) :
        mu = K.mean( x, axis=(1,2), keepdims=True )
        if self.scaling :
            sigma = K.maximum( K.std( x, axis=(1,2), keepdims=True ), self.min_std )
            xn = ( x-mu ) / sigma
            if ( self.mode == 'thresh' ) :
                diff = K.minimum( K.maximum( xn, -self.min_dev), self.max_dev )
            elif ( self.mode == 'tanh' ) :
                diff = K.tanh( diff )
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

def bn_conv2d( x, nfilters, kernal_size, name, use_bn=False, use_prelu=False, kernel_regularizer=None ) :
    '''basic Conv2D+BatchNorm+ReLu module
    '''
    x = Conv2DSymPadding( nfilters,
                          kernal_size,
                          padding='same',
                          activation=None,
                          kernel_regularizer=kernel_regularizer,
                          name=name+'-conv' )(x)
    if ( use_bn ) :
        x = BatchNormalization( name=name+'-bnorm' )(x)
    if ( use_prelu ) :
        x = PReLU(shared_axes=[1,2], name=name+'-prelu')(x)
    else :
        x = Activation('relu', name=name+'-relu' )(x)
    return x

def create_model( input_shape=(None,None,3),
                  name='SignatureNet',
                  nb_outputs=1,
                  nb_layers=17,
                  use_bn=True,
                  use_prelu=False,
                  deviation_type='Euclidean',
                  centering=False,
                  scaling=False,
                  decision_size=(7,7)) :
    '''SignatureNet
    '''
    obs = Input( shape=input_shape, name=name+'-input' )
    x = Conv2DSymPadding( 64, (3,3),
                activation='relu',
                name=name+'-c1',
                padding='same' )( obs )
    for k in range(nb_layers-2) :
        x = bn_conv2d( x, 64, (3,3),
                       name = name +'-c%d' % (k+2),
                       use_bn=use_bn,
                       use_prelu=use_prelu )
    # compute local statistics
    if ( deviation_type == 'Euclidean' ) :
        Deviation = EuclideanDeviation
    else :
        Deviation = AngularDeviation
    x = Deviation(centering=centering,
                  scaling=scaling,
                  name='normalize')(x)
    # decision
    if ( not scaling ) :
        x = BatchNormalization( name=name+'mbnorm')(x)
    y = Conv2DSymPadding( nb_outputs, decision_size,
                activation='sigmoid',
                name=name+'-mask',
                padding='same' )(x)
    return Model( inputs=obs, output=y, name=name )
