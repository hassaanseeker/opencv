import tensorflow as tf
import numpy as np

def getStrides(layer):
    dx = layer.getLayerParam_int('stride_w', 0)
    dy = layer.getLayerParam_int('stride_h', 0)
    if dx == 0 or dy == 0:
        assert(dx == 0 and dy == 0)
        dx = layer.getLayerParam_int('stride', 1)
        return dx, dx
    else:
        return dx, dy

def getKernelSize(layer):
    kw = layer.getLayerParam_int('kernel_w', 0)
    kh = layer.getLayerParam_int('kernel_h', 0)
    if kw == 0 or kh == 0:
        assert(kw == 0 and kh == 0)
        kw = layer.getLayerParam_int('kernel', 0)
        if kw == 0:
            kw = layer.getLayerParam_int('kernel_size', 0)
        return kw, kw
    else:
        return kw, kh

def getPaddingMode(layer):
    if layer.getLayerParam_int('pad_w', 0) != 0 or \
       layer.getLayerParam_int('pad_h', 0) != 0 or \
       layer.getLayerParam_int('pad', 0) != 0:
       return 'SAME'
    else:
       return 'VALID'

# Converts DNN layer to TensorFlow node
def convert(layer, x, dtype):
    ############################################################################
    # Parametric layers (nas weights)
    ############################################################################
    if layer.type == 'Convolution':
        dx, dy = getStrides(layer)
        padMode = getPaddingMode(layer)

        w = tf.constant(layer.blobs[0].transpose(2, 3, 1, 0), dtype=dtype, name=layer.name + '_weights')
        conv = tf.nn.conv2d(x, filter=w, strides=(1, dy, dx, 1), padding=padMode, name=layer.name)

        if len(layer.blobs) > 1:
            b = tf.constant(layer.blobs[1].flatten(), dtype=dtype, name=layer.name + '_bias')
            conv = tf.nn.bias_add(conv, b)
        return conv

    elif layer.type == 'InnerProduct':
        # w = tf.constant(layer.blobs[0].transpose(2, 3, 1, 0))
        # conv = tf.nn.conv2d(x, filter=w,
        #                     strides=(1, dy, dx, 1), padding=padMode)
        #
        # if len(layer.blobs) > 1:
        #     b = tf.constant(layer.blobs[1].flatten())
        #     conv = tf.nn.bias_add(conv, b)
        return None
    ############################################################################
    # Non-parametric layers (has no weights)
    ############################################################################
    elif layer.type == 'ReLU':
        return tf.nn.relu(x, layer.name)

    elif layer.type == 'LRN':
        bias = layer.getLayerParam_float('bias')
        alpha = layer.getLayerParam_float('alpha')
        beta = layer.getLayerParam_float('beta')
        norm_by_size = layer.getLayerParam_bool('norm_by_size')
        local_size = layer.getLayerParam_int('local_size')
        if norm_by_size:
            alpha /= local_size
        return tf.nn.local_response_normalization(x, local_size, bias, alpha, beta, layer.name)

    elif layer.type == 'Pooling':
        pool = layer.getLayerParam_str('pool')
        kw, kh = getKernelSize(layer)
        dx, dy = getStrides(layer)
        if (x.shape[1] - kh) % dy != 0 or (x.shape[2] - kw) % dx != 0:
            padMode = 'SAME'
        else:
            padMode = 'VALID'
        # padMode = getPaddingMode(layer)

        if pool == 'MAX':
            assert(kw != 0 and kh != 0)
            return tf.layers.max_pooling2d(x, pool_size=(kh, kw), strides=(dy, dx),
                                           padding=padMode, name=layer.name)
        else:
            assert(pool == 'AVE')
            if kw == 0 and kh == 0:
                kw = x.shape[2]
                kh = x.shape[1]
            return tf.layers.average_pooling2d(x, pool_size=(kh, kw), strides=(dy, dx),
                                               padding=padMode, name=layer.name)

    elif layer.type == 'Concat':
        # axis = layer.getLayerParam_int('axis', 1)
        return tf.concat(x, 3, layer.name)

    elif layer.type == 'Flatten':
        shape = int(np.prod(x.shape))
        return tf.reshape(x, [-1, shape], layer.name)

    elif layer.type == 'Dropout':
        return x

    elif layer.type == 'Softmax':
        return tf.nn.softmax(x, name=layer.name)

    else:
        print('Unknown layer type ' + layer.type)
        assert(False)
