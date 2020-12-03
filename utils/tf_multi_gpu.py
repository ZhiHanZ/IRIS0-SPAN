from keras.layers import merge, Concatenate
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf
def make_parallel(model, gpu_count, output_names = None, gpu_list = None ):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])
    if gpu_list is None :
        gpu_list = range( gpu_count )
    #Place a copy of the model on each GPU, each getting a slice of the batch
    for j, i in enumerate(gpu_list):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':j,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        if ( output_names is None ) :
            output_names = [ 'out-%d' % v for v in range( len( outputs_all ) ) ]
        #try :
        #    for name, outputs in zip( output_names, outputs_all ):
        #        merged.append(merge(outputs, mode='concat', concat_axis=0, name = name ) )
        #except :
        for name, outputs in zip( output_names, outputs_all ):
            merged.append(Concatenate(axis=0, name = name )(outputs) )

        return Model(inputs=model.inputs, outputs=merged)
