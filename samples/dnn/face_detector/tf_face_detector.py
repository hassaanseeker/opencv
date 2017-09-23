import argparse
import cv2 as cv
import tensorflow as tf
import numpy as np
import struct

from tensorflow.python.framework import function

parser = argparse.ArgumentParser(description='TensorFlow graph definition for'
                                             'OpenCV face detection model.')
parser.add_argument('-m', dest='caffemodel', required=True,
                    help='Path to .caffemodel weights')
parser.add_argument('-p', dest='prototxt', required=True,
                    help='Path to .prototxt Caffe model definition')
args = parser.parse_args()

dtype = tf.float32

### Import net into DNN ########################################################
cvNet = cv.dnn.readNetFromCaffe(args.prototxt, args.caffemodel)

def dnnLayer(name):
    return cvNet.getLayer(long(cvNet.getLayerId(name)))

def scale(x, name):
    layer = dnnLayer(name)
    w = tf.Variable(layer.blobs[0].flatten(), dtype=dtype)
    if len(layer.blobs) > 1:
        b = tf.Variable(layer.blobs[1].flatten(), dtype=dtype)
        return tf.nn.bias_add(tf.multiply(x, w), b, name=name)
    else:
        return tf.multiply(x, w, name)

def conv(x, name, stride=1, pad='SAME', dilation=1):
    layer = dnnLayer(name)
    w = tf.Variable(layer.blobs[0].transpose(2, 3, 1, 0), dtype=dtype)
    if dilation == 1:
        conv = tf.nn.conv2d(x, filter=w, strides=(1, stride, stride, 1), padding=pad, name=name)
    else:
        assert(stride == 1)
        conv = tf.nn.atrous_conv2d(x, w, rate=dilation, padding=pad, name=name)

    if len(layer.blobs) > 1:
        b = tf.Variable(layer.blobs[1].flatten(), dtype=dtype)
        conv = tf.nn.bias_add(conv, b, name=name)
    return conv

def batch_norm(x, name):
    # Unfortunately, TensorFlow's batch normalization layer doesn't work with fp16 input.
    # Here we do a cast to fp32 but remove it in the frozen graph.
    if x.dtype != tf.float32:
        x = tf.cast(x, tf.float32)

    layer = dnnLayer(name)
    assert(len(layer.blobs) >= 3)

    mean = layer.blobs[0].flatten()
    std = layer.blobs[1].flatten()
    scale = layer.blobs[2].flatten()

    eps = 1e-5
    hasBias = len(layer.blobs) > 3
    hasWeights = scale.shape != (1,)

    if not hasWeights and not hasBias:
        mean /= scale[0]
        std /= scale[0]

    mean = tf.Variable(mean, dtype=tf.float32)
    std = tf.Variable(std, dtype=tf.float32)
    gamma = tf.Variable(scale if hasWeights else np.ones(mean.shape), dtype=tf.float32)
    beta = tf.Variable(layer.blobs[3].flatten() if hasBias else np.zeros(mean.shape), dtype=tf.float32)
    bn = tf.nn.fused_batch_norm(x, gamma, beta, mean, std, eps,
                                is_training=False, name=name)[0]
    if bn.dtype != dtype:
        bn = tf.cast(bn, dtype)
    return bn

@function.Defun(dtype, dtype, func_name='L2Normalize',
                shape_func=lambda op: [op.inputs[0].shape])
def L2NormalizeLayer(x, w):
    return tf.nn.l2_normalize(x, 3, epsilon=1e-10) * w

def l2norm(x, name):
    layer = dnnLayer(name)
    w = tf.Variable(layer.blobs[0].flatten(), dtype=dtype)
    return L2NormalizeLayer(x, w)

### Graph definition ###########################################################
inp = tf.placeholder(dtype, [None, 300, 300, 3], 'data')
data_bn = batch_norm(inp, 'data_bn')
data_scale = scale(data_bn, 'data_scale')
data_scale = tf.pad(data_scale, [[0, 0], [3, 3], [3, 3], [0, 0]])
conv1_h = conv(data_scale, stride=2, pad='VALID', name='conv1_h')
conv1_bn_h = batch_norm(conv1_h, 'conv1_bn_h')
conv1_scale_h = scale(conv1_bn_h, 'conv1_scale_h')
conv1_relu = tf.nn.relu(conv1_scale_h)
conv1_pool = tf.layers.max_pooling2d(conv1_relu, pool_size=(3, 3), strides=(2, 2),
                                     padding='SAME', name='conv1_pool')

layer_64_1_conv1_h = conv(conv1_pool, 'layer_64_1_conv1_h')
layer_64_1_bn2_h = batch_norm(layer_64_1_conv1_h, 'layer_64_1_bn2_h')
layer_64_1_scale2_h = scale(layer_64_1_bn2_h, 'layer_64_1_scale2_h')
layer_64_1_relu2 = tf.nn.relu(layer_64_1_scale2_h)
layer_64_1_conv2_h = conv(layer_64_1_relu2, 'layer_64_1_conv2_h')
layer_64_1_sum = layer_64_1_conv2_h + conv1_pool

layer_128_1_bn1_h = batch_norm(layer_64_1_sum, 'layer_128_1_bn1_h')
layer_128_1_scale1_h = scale(layer_128_1_bn1_h, 'layer_128_1_scale1_h')
layer_128_1_relu1 = tf.nn.relu(layer_128_1_scale1_h)
layer_128_1_conv1_h = conv(layer_128_1_relu1, stride=2, name='layer_128_1_conv1_h')
layer_128_1_bn2 = batch_norm(layer_128_1_conv1_h, 'layer_128_1_bn2')
layer_128_1_scale2 = scale(layer_128_1_bn2, 'layer_128_1_scale2')
layer_128_1_relu2 = tf.nn.relu(layer_128_1_scale2)
layer_128_1_conv2 = conv(layer_128_1_relu2, 'layer_128_1_conv2')
layer_128_1_conv_expand_h = conv(layer_128_1_relu1, stride=2, name='layer_128_1_conv_expand_h')
layer_128_1_sum = layer_128_1_conv2 + layer_128_1_conv_expand_h

layer_256_1_bn1 = batch_norm(layer_128_1_sum, 'layer_256_1_bn1')
layer_256_1_scale1 = scale(layer_256_1_bn1, 'layer_256_1_scale1')
layer_256_1_relu1 = tf.nn.relu(layer_256_1_scale1)
layer_256_1_conv1 = tf.pad(layer_256_1_relu1, [[0, 0], [1, 1], [1, 1], [0, 0]])
layer_256_1_conv1 = conv(layer_256_1_conv1, stride=2, pad='VALID', name='layer_256_1_conv1')
layer_256_1_bn2 = batch_norm(layer_256_1_conv1, 'layer_256_1_bn2')
layer_256_1_scale2 = scale(layer_256_1_bn2, 'layer_256_1_scale2')
layer_256_1_relu2 = tf.nn.relu(layer_256_1_scale2)
layer_256_1_conv2 = conv(layer_256_1_relu2, 'layer_256_1_conv2')
layer_256_1_conv_expand = conv(layer_256_1_relu1, stride=2, name='layer_256_1_conv_expand')
layer_256_1_sum = layer_256_1_conv2 + layer_256_1_conv_expand

layer_512_1_bn1 = batch_norm(layer_256_1_sum, 'layer_512_1_bn1')
layer_512_1_scale1 = scale(layer_512_1_bn1, 'layer_512_1_scale1')
layer_512_1_relu1 = tf.nn.relu(layer_512_1_scale1)
layer_512_1_conv1_h = conv(layer_512_1_relu1, 'layer_512_1_conv1_h')
layer_512_1_bn2_h = batch_norm(layer_512_1_conv1_h, 'layer_512_1_bn2_h')
layer_512_1_scale2_h = scale(layer_512_1_bn2_h, 'layer_512_1_scale2_h')
layer_512_1_relu2 = tf.nn.relu(layer_512_1_scale2_h)
layer_512_1_conv2_h = conv(layer_512_1_relu2, dilation=2, name='layer_512_1_conv2_h')
layer_512_1_conv_expand_h = conv(layer_512_1_relu1, 'layer_512_1_conv_expand_h')
layer_512_1_sum = layer_512_1_conv2_h + layer_512_1_conv_expand_h

last_bn_h = batch_norm(layer_512_1_sum, 'last_bn_h')
last_scale_h = scale(last_bn_h, 'last_scale_h')
fc7 = tf.nn.relu(last_scale_h)

conv6_1_h = tf.nn.relu(conv(fc7, 'conv6_1_h'))
conv6_2_h = tf.nn.relu(conv(conv6_1_h, stride=2, name='conv6_2_h'))
conv7_1_h = tf.nn.relu(conv(conv6_2_h, 'conv7_1_h'))
conv7_2_h = tf.pad(conv7_1_h, [[0, 0], [1, 1], [1, 1], [0, 0]])
conv7_2_h = tf.nn.relu(conv(conv7_2_h, stride=2, pad='VALID', name='conv7_2_h'))
conv8_1_h = tf.nn.relu(conv(conv7_2_h, pad='SAME', name='conv8_1_h'))
conv8_2_h = tf.nn.relu(conv(conv8_1_h, pad='VALID', name='conv8_2_h'))
conv9_1_h = tf.nn.relu(conv(conv8_2_h, 'conv9_1_h'))
conv9_2_h = tf.nn.relu(conv(conv9_1_h, pad='VALID', name='conv9_2_h'))

conv4_3_norm = l2norm(layer_256_1_relu1, 'conv4_3_norm')

### Locations ##################################################################
conv4_3_norm_mbox_loc = conv(conv4_3_norm, 'conv4_3_norm_mbox_loc')
# Permute to NHWC layout. Here it's an identity op but it's require for DNN.
conv4_3_norm_mbox_loc_perm = tf.transpose(conv4_3_norm_mbox_loc, [0, 1, 2, 3])
total = int(np.prod(conv4_3_norm_mbox_loc_perm.shape[1:]))
conv4_3_norm_mbox_loc_flat = tf.reshape(conv4_3_norm_mbox_loc_perm, [-1, total])

fc7_mbox_loc = conv(fc7, 'fc7_mbox_loc')
fc7_mbox_loc_perm = tf.transpose(fc7_mbox_loc, [0, 1, 2, 3])
total = int(np.prod(fc7_mbox_loc_perm.shape[1:]))
fc7_mbox_loc_flat = tf.reshape(fc7_mbox_loc_perm, [-1, total])

conv6_2_mbox_loc = conv(conv6_2_h, 'conv6_2_mbox_loc')
conv6_2_mbox_loc_perm = tf.transpose(conv6_2_mbox_loc, [0, 1, 2, 3])
total = int(np.prod(conv6_2_mbox_loc_perm.shape[1:]))
conv6_2_mbox_loc_flat = tf.reshape(conv6_2_mbox_loc_perm, [-1, total])

conv7_2_mbox_loc = conv(conv7_2_h, 'conv7_2_mbox_loc')
conv7_2_mbox_loc_perm = tf.transpose(conv7_2_mbox_loc, [0, 1, 2, 3])
total = int(np.prod(conv7_2_mbox_loc_perm.shape[1:]))
conv7_2_mbox_loc_flat = tf.reshape(conv7_2_mbox_loc_perm, [-1, total])

conv8_2_mbox_loc = conv(conv8_2_h, 'conv8_2_mbox_loc')
conv8_2_mbox_loc_perm = tf.transpose(conv8_2_mbox_loc, [0, 1, 2, 3])
total = int(np.prod(conv8_2_mbox_loc_perm.shape[1:]))
conv8_2_mbox_loc_flat = tf.reshape(conv8_2_mbox_loc_perm, [-1, total])

conv9_2_mbox_loc = conv(conv9_2_h, 'conv9_2_mbox_loc')
conv9_2_mbox_loc_perm = tf.transpose(conv9_2_mbox_loc, [0, 1, 2, 3])
total = int(np.prod(conv9_2_mbox_loc_perm.shape[1:]))
conv9_2_mbox_loc_flat = tf.reshape(conv9_2_mbox_loc_perm, [-1, total])

mbox_loc = tf.concat([conv4_3_norm_mbox_loc_flat,
                      fc7_mbox_loc_flat,
                      conv6_2_mbox_loc_flat,
                      conv7_2_mbox_loc_flat,
                      conv8_2_mbox_loc_flat,
                      conv9_2_mbox_loc_flat], axis=-1, name='mbox_loc')

### Confidences ################################################################
conv4_3_norm_mbox_conf = conv(conv4_3_norm, 'conv4_3_norm_mbox_conf')
conv4_3_norm_mbox_conf_perm = tf.transpose(conv4_3_norm_mbox_conf, [0, 1, 2, 3])
total = int(np.prod(conv4_3_norm_mbox_conf_perm.shape[1:]))
conv4_3_norm_mbox_conf_flat = tf.reshape(conv4_3_norm_mbox_conf_perm, [-1, total])

fc7_mbox_conf = conv(fc7, 'fc7_mbox_conf')
fc7_mbox_conf_perm = tf.transpose(fc7_mbox_conf, [0, 1, 2, 3])
total = int(np.prod(fc7_mbox_conf_perm.shape[1:]))
fc7_mbox_conf_flat = tf.reshape(fc7_mbox_conf_perm, [-1, total])

conv6_2_mbox_conf = conv(conv6_2_h, 'conv6_2_mbox_conf')
conv6_2_mbox_conf_perm = tf.transpose(conv6_2_mbox_conf, [0, 1, 2, 3])
total = int(np.prod(conv6_2_mbox_conf_perm.shape[1:]))
conv6_2_mbox_conf_flat = tf.reshape(conv6_2_mbox_conf_perm, [-1, total])

conv7_2_mbox_conf = conv(conv7_2_h, 'conv7_2_mbox_conf')
conv7_2_mbox_conf_perm = tf.transpose(conv7_2_mbox_conf, [0, 1, 2, 3])
total = int(np.prod(conv7_2_mbox_conf_perm.shape[1:]))
conv7_2_mbox_conf_flat = tf.reshape(conv7_2_mbox_conf_perm, [-1, total])

conv8_2_mbox_conf = conv(conv8_2_h, 'conv8_2_mbox_conf')
conv8_2_mbox_conf_perm = tf.transpose(conv8_2_mbox_conf, [0, 1, 2, 3])
total = int(np.prod(conv8_2_mbox_conf_perm.shape[1:]))
conv8_2_mbox_conf_flat = tf.reshape(conv8_2_mbox_conf_perm, [-1, total])

conv9_2_mbox_conf = conv(conv9_2_h, 'conv9_2_mbox_conf')
conv9_2_mbox_conf_perm = tf.transpose(conv9_2_mbox_conf, [0, 1, 2, 3])
total = int(np.prod(conv9_2_mbox_conf_perm.shape[1:]))
conv9_2_mbox_conf_flat = tf.reshape(conv9_2_mbox_conf_perm, [-1, total])

mbox_conf = tf.concat([conv4_3_norm_mbox_conf_flat,
                      fc7_mbox_conf_flat,
                      conv6_2_mbox_conf_flat,
                      conv7_2_mbox_conf_flat,
                      conv8_2_mbox_conf_flat,
                      conv9_2_mbox_conf_flat], axis=-1, name='mbox_conf')

total = int(np.prod(mbox_conf.shape[1:]))
mbox_conf_reshape = tf.reshape(mbox_conf, [-1, 2], name='mbox_conf_reshape')
mbox_conf_softmax = tf.nn.softmax(mbox_conf_reshape, name='mbox_conf_softmax')
mbox_conf_flatten = tf.reshape(mbox_conf_softmax, [-1, total], name='mbox_conf_flatten')

### Prior boxes ################################################################
numPriors = 2
@function.Defun(dtype, dtype, tf.int32, tf.int32, tf.float32, tf.bool, tf.bool,
                tf.float32, tf.int32, tf.float32, func_name='PriorBox',
                shape_func=lambda op: [[1, 2, op.inputs[0].shape[1] * op.inputs[0].shape[2] * numPriors * 4]])
def priorBoxLayer(x, y, minSize, maxSize, aspectRatios, flip, clip, variance, step, offset):
    # Fake op.
    return tf.zeros([1], dtype=dtype)

def priorBox(x, y, minSize, maxSize, aspectRatios, flip, clip, variance, step, offset):
    return priorBoxLayer(x, y, minSize, maxSize, aspectRatios, flip, clip, variance, step, offset)

conv4_3_norm_mbox_priorbox = priorBox(conv4_3_norm, inp, minSize=30, maxSize=60,
                                      aspectRatios=[2.], flip=True, clip=False,
                                      variance=[0.1, 0.1, 0.2, 0.2], step=8, offset=0.5)

fc7_mbox_priorbox = priorBox(fc7, inp, minSize=60, maxSize=111,
                             aspectRatios=[2., 3.], flip=True, clip=False,
                             variance=[0.1, 0.1, 0.2, 0.2], step=16, offset=0.5)

conv6_2_mbox_priorbox = priorBox(conv6_2_h, inp, minSize=111, maxSize=162,
                                 aspectRatios=[2., 3.], flip=True, clip=False,
                                 variance=[0.1, 0.1, 0.2, 0.2], step=32, offset=0.5)

conv7_2_mbox_priorbox = priorBox(conv7_2_h, inp, minSize=162, maxSize=213,
                                 aspectRatios=[2., 3.], flip=True, clip=False,
                                 variance=[0.1, 0.1, 0.2, 0.2], step=64, offset=0.5)

conv8_2_mbox_priorbox = priorBox(conv8_2_h, inp, minSize=213, maxSize=264,
                                 aspectRatios=[2.], flip=True, clip=False,
                                 variance=[0.1, 0.1, 0.2, 0.2], step=100, offset=0.5)

conv9_2_mbox_priorbox = priorBox(conv9_2_h, inp, minSize=264, maxSize=315,
                                 aspectRatios=[2.], flip=True, clip=False,
                                 variance=[0.1, 0.1, 0.2, 0.2], step=300, offset=0.5)

mbox_priorbox = tf.concat([conv4_3_norm_mbox_priorbox,
                           fc7_mbox_priorbox,
                           conv6_2_mbox_priorbox,
                           conv7_2_mbox_priorbox,
                           conv8_2_mbox_priorbox,
                           conv9_2_mbox_priorbox], axis=-1)

### Detection output layer #####################################################
@function.Defun(dtype, dtype, dtype, tf.int32, tf.bool, tf.int32,
                tf.float32, tf.int32, tf.string, tf.int32, tf.float32, func_name='DetectionOutput')
def detectionOutputLayer(loc, conf, priorBoxes, num_classes, share_location, background_label_id,
                         nms_threshold, top_k, code_type, keep_top_k, confidence_threshold):
    # Fake op.
    return tf.zeros([1], dtype=dtype)

def detectionOutput(loc, conf, priorBoxes, num_classes, share_location, background_label_id,
                         nms_threshold, top_k, code_type, keep_top_k, confidence_threshold):
    return detectionOutputLayer(loc, conf, priorBoxes, num_classes, share_location, background_label_id,
                                nms_threshold, top_k, code_type, keep_top_k, confidence_threshold)

detection_out = detectionOutput(mbox_loc, mbox_conf_flatten, mbox_priorbox, num_classes=2,
                                share_location=True, background_label_id=0, nms_threshold=0.45,
                                top_k=400, code_type='CENTER_SIZE', keep_top_k=200, confidence_threshold=0.01)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ### Save graph #############################################################
    saver = tf.train.Saver()
    saver.save(sess, 'face_detector.ckpt')

    # By default, float16 weights are stored in repeated tensor's field called
    # `half_val`. It has type int32 with leading zeros for unused bytes.
    # This type is encoded by Varint that means only 7 bits are used for value
    # representation but the last one is indicated the end of encoding. This way
    # float16 might takes 1 or 2 or 3 bytes depends on value. To impove compression,
    # we replace all `half_val` values to `tensor_content` using only 2 bytes for everyone.
    graph_def = sess.graph.as_graph_def()
    for node in graph_def.node:
        if 'value' in node.attr:
            halfs = node.attr["value"].tensor.half_val
            if not node.attr["value"].tensor.tensor_content and halfs:
                node.attr["value"].tensor.tensor_content = struct.pack('H' * len(halfs), *halfs)
                node.attr["value"].tensor.ClearField('half_val')
    tf.train.write_graph(graph_def, "", 'face_detector.pb')

    ## Check correctness ######################################################
    np.random.seed(2701)
    inputData = np.random.standard_normal([1, 3, 300, 300]).astype(np.float32)

    cvNet.setInput(inputData)
    outDNN = cvNet.forward(['mbox_loc', 'mbox_conf_flatten'])

    outTF = sess.run([mbox_loc, mbox_conf_flatten], feed_dict={inp: inputData.transpose(0, 2, 3, 1)})
    print 'Max diff @ locations:  %e' % np.max(np.abs(outDNN[0] - outTF[0]))
    print 'Max diff @ confidence: %e' % np.max(np.abs(outDNN[1] - outTF[1]))
