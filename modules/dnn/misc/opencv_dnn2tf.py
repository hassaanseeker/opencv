import cv2 as cv
import tensorflow as tf
import argparse
import numpy as np
from convert_layer import convert

parser = argparse.ArgumentParser(description='This script converts networks from '
                                 'the OpenCV\'s DNN internal representation to '
                                 'TensorFlow graph')
parser.add_argument('-p', dest='prototxt', help='Path to .prototxt of Caffe net')
parser.add_argument('-c', dest='caffemodel', help='Path to .caffemodel of Caffe net')
parser.add_argument('--width', help='Width of input image')
parser.add_argument('--height', help='Height of input image')
args = parser.parse_args()

w = int(args.width)
h = int(args.height)

inputData = np.random.standard_normal([1, 3, h, w]).astype(np.float32)

# test_layer_name = 'prob'

cvNet = cv.dnn.readNetFromCaffe(args.prototxt, args.caffemodel)
cvNet.setInput(inputData)
outCaffeNet = cvNet.forward('prob')

layerNames = cvNet.getLayerNames()

tf.reset_default_graph()
tf.Graph().as_default()
sess = tf.Session()

dtype = tf.float32
layers = [tf.placeholder(dtype, [1, h, w, 3], 'input')]

for i in range(len(layerNames)):
    i += 1  # Skip input layer id.
    layer = cvNet.getLayer(long(i))

    # print i, layer.name, layer.type

    inputs = cvNet.getLayerInputs(long(i))
    inputs = [layers[idx[0]] for idx in inputs]

    if len(inputs) == 1:
        layer = convert(layer, inputs[0], dtype)
    else:
        layer = convert(layer, inputs, dtype)
    layers.append(layer)

    # if cvNet.getLayer(long(i)).name == test_layer_name:
    #     break

outTF = sess.run(layers[-1], feed_dict={layers[0]: inputData.transpose(0, 2, 3, 1)})

print np.max(np.abs(outCaffeNet - outTF))

tf.train.write_graph(sess.graph.as_graph_def(), "", "graph.pb", as_text=False)

cvNet = cv.dnn.readNetFromTensorflow("graph.pb")
cvNet.setInput(inputData)
print np.max(np.abs(outCaffeNet - cvNet.forward('prob')))
