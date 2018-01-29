import tensorflow as tf
import argparse
from math import sqrt

parser = argparse.ArgumentParser(description='Run this script to get text graph of '
                                             'SSD model from TensorFlow Object Detection API. '
                                             'Then pass it with .pb file to cv::dnn::readNetFromTensorflow function.')
parser.add_argument('--input', required=True, help='Path to frozen TensorFlow graph.')
parser.add_argument('--output', required=True, help='Path to output text graph.')
parser.add_argument('--num_classes', default=90, type=int, help='Number of trained classes.')
parser.add_argument('--min_scale', default=0.2, type=float, help='Hyper-parameter of ssd_anchor_generator from config file.')
parser.add_argument('--max_scale', default=0.95, type=float, help='Hyper-parameter of ssd_anchor_generator from config file.')
parser.add_argument('--num_layers', default=6, type=int, help='Hyper-parameter of ssd_anchor_generator from config file.')
parser.add_argument('--aspect_ratios', default=[1.0, 2.0, 0.5, 3.0, 0.333], type=float, nargs='+',
                    help='Hyper-parameter of ssd_anchor_generator from config file.')
args = parser.parse_args()

# Nodes that should be kept.
keepOps = ['Conv2D', 'BiasAdd', 'Add', 'Relu6', 'Placeholder', 'FusedBatchNorm',
           'DepthwiseConv2dNative', 'ConcatV2', 'Mul', 'MaxPool', 'AvgPool']

# Nodes attributes that could be removed because they are not used during import.
unusedAttrs = ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim', 'use_cudnn_on_gpu',
               'Index', 'Tperm', 'is_training', 'Tpaddings']

# Node with which prefixes should be removed
prefixesToRemove = ('MultipleGridAnchorGenerator/', 'Postprocessor/', 'Preprocessor/')

# Read the graph.
with tf.gfile.FastGFile(args.input, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

def getUnconnectedNodes():
    unconnected = []
    for node in graph_def.node:
        unconnected.append(node.name)
        for inp in node.input:
            if inp in unconnected:
                unconnected.remove(inp)
    return unconnected

removedNodes = []

# Remove extra nodes and attributes.
for i in reversed(range(len(graph_def.node))):
    op = graph_def.node[i].op
    name = graph_def.node[i].name

    if (not op in keepOps) or name.startswith(prefixesToRemove):
        if op != 'Const':
            removedNodes.append(name)

        del graph_def.node[i]
    else:
        for attr in unusedAttrs:
            if attr in graph_def.node[i].attr:
                del graph_def.node[i].attr[attr]

# Remove references to removed nodes except Const nodes.
for node in graph_def.node:
    for i in reversed(range(len(node.input))):
        if node.input[i] in removedNodes:
            del node.input[i]

# Connect input node to the first layer
assert(graph_def.node[0].op == 'Placeholder')
# assert(graph_def.node[1].op == 'Conv2D')
weights = graph_def.node[1].input[0]
for i in range(len(graph_def.node[1].input)):
    graph_def.node[1].input.pop()
graph_def.node[1].input.append(graph_def.node[0].name)
graph_def.node[1].input.append(weights)

# Create SSD postprocessing head ###############################################
from tensorflow.core.framework.node_def_pb2 import NodeDef
from google.protobuf import text_format

# Concatenate predictions of classes, predictions of bounding boxes and proposals.

concatAxis = NodeDef()
concatAxis.name = 'concat/axis_flatten'
concatAxis.op = 'Const'
text_format.Merge(
'tensor {'
'  dtype: DT_INT32'
'  tensor_shape { }'
'  int_val: -1'
'}', concatAxis.attr["value"])
graph_def.node.extend([concatAxis])

def addConcatNode(name, inputs):
    concat = NodeDef()
    concat.name = name
    concat.op = 'ConcatV2'
    for inp in inputs:
        concat.input.append(inp)
    concat.input.append(concatAxis.name)
    graph_def.node.extend([concat])

for label in ['ClassPredictor', 'BoxEncodingPredictor']:
    concatInputs = []
    for i in range(args.num_layers):
        # Flatten predictions
        flatten = NodeDef()
        inpName = 'BoxPredictor_%d/%s/BiasAdd' % (i, label)
        flatten.input.append(inpName)
        flatten.name = inpName + '/Flatten'
        flatten.op = 'Flatten'

        concatInputs.append(flatten.name)
        graph_def.node.extend([flatten])
    addConcatNode('%s/concat' % label, concatInputs)

# Add layers that generate anchors (bounding boxes proposals).
scales = [args.min_scale + (args.max_scale - args.min_scale) * i / (args.num_layers - 1)
          for i in range(args.num_layers)] + [1.0]

def tensorMsg(values):
    msg = 'tensor { dtype: DT_FLOAT tensor_shape { dim { size: %d } }' % len(values)
    for value in values:
        msg += 'float_val: %f ' % value
    return msg + '}'

priorBoxes = []
for i in range(args.num_layers):
    priorBox = NodeDef()
    priorBox.name = 'PriorBox_%d' % i
    priorBox.op = 'PriorBox'
    priorBox.input.append('BoxPredictor_%d/BoxEncodingPredictor/BiasAdd' % i)
    priorBox.input.append(graph_def.node[0].name)  # image_tensor

    text_format.Merge('b: false', priorBox.attr["flip"])
    text_format.Merge('b: false', priorBox.attr["clip"])
    text_format.Merge('b: true', priorBox.attr["normalized_bbox"])

    if i == 0:
        widths = [args.min_scale * 0.5, args.min_scale * sqrt(2.0), args.min_scale * sqrt(0.5)]
        heights = [args.min_scale * 0.5, args.min_scale / sqrt(2.0), args.min_scale / sqrt(0.5)]
    else:
        widths = [scales[i] * sqrt(ar) for ar in args.aspect_ratios]
        heights = [scales[i] / sqrt(ar) for ar in args.aspect_ratios]

        widths += [sqrt(scales[i] * scales[i + 1])]
        heights += [sqrt(scales[i] * scales[i + 1])]
    text_format.Merge(tensorMsg(widths), priorBox.attr["width"])
    text_format.Merge(tensorMsg(heights), priorBox.attr["height"])
    text_format.Merge(tensorMsg([0.1, 0.1, 0.2, 0.2]), priorBox.attr["variance"])

    graph_def.node.extend([priorBox])
    priorBoxes.append(priorBox.name)

addConcatNode('PriorBox/concat', priorBoxes)

# Sigmoid for classes predictions and DetectionOutput layer
sigmoid = NodeDef()
sigmoid.name = 'ClassPredictor/concat/sigmoid'
sigmoid.op = 'Sigmoid'
sigmoid.input.append('ClassPredictor/concat')
graph_def.node.extend([sigmoid])

detectionOut = NodeDef()
detectionOut.name = 'detection_out'
detectionOut.op = 'DetectionOutput'

detectionOut.input.append('BoxEncodingPredictor/concat')
detectionOut.input.append(sigmoid.name)
detectionOut.input.append('PriorBox/concat')

text_format.Merge('i: %d' % (args.num_classes + 1), detectionOut.attr['num_classes'])
text_format.Merge('b: true', detectionOut.attr['share_location'])
text_format.Merge('i: 0', detectionOut.attr['background_label_id'])
text_format.Merge('f: 0.6', detectionOut.attr['nms_threshold'])
text_format.Merge('i: 100', detectionOut.attr['top_k'])
text_format.Merge('s: "CENTER_SIZE"', detectionOut.attr['code_type'])
text_format.Merge('i: 100', detectionOut.attr['keep_top_k'])
text_format.Merge('f: 0.01', detectionOut.attr['confidence_threshold'])
text_format.Merge('b: true', detectionOut.attr['loc_pred_transposed'])

graph_def.node.extend([detectionOut])

while True:
    unconnectedNodes = getUnconnectedNodes()
    unconnectedNodes.remove(detectionOut.name)
    if not unconnectedNodes:
        break

    for name in unconnectedNodes:
        for i in range(len(graph_def.node)):
            if graph_def.node[i].name == name:
                del graph_def.node[i]
                break

# Save as text.
tf.train.write_graph(graph_def, "", args.output, as_text=True)
